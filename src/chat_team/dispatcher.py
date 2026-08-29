"""Dispatcher: glues IncomingMessage → Session → Agent → Response.

Owns transfer handling (per-turn cap, handoff_note injection, re-invoking
the new agent on the original user message).
"""
from __future__ import annotations

import asyncio
import logging

from .adapters.base import IncomingMessage, StreamHandle
from .agent.agent import Agent
from .agent.compactor import maybe_compact
from .agent.tools.base import ToolRegistry, TransferRequested
from .config import Settings
from .llm.base import LLMProvider
from .roles.registry import RoleRegistry
from .session.manager import SessionManager
from .session.persistence import PersistenceManager
from .session.session import PendingHandoff, Session
from .skills.registry import SkillRegistry
from .vision_shim import apply_vision_strategy

log = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        sessions: SessionManager,
        roles: RoleRegistry,
        tools: ToolRegistry,
        llm: LLMProvider,
        skills: SkillRegistry | None = None,
        persistence: PersistenceManager | None = None,
        vision_llm: LLMProvider | None = None,
        fixed_role: str | None = None,
    ):
        self.settings = settings
        self.sessions = sessions
        self.roles = roles
        self.tools = tools
        self.llm = llm
        self.skills = skills if skills is not None else SkillRegistry({})
        self.persistence = persistence
        # Separate provider for vision/OCR calls (describe_image tool).
        # Falls back to self.llm when not set.
        self._vision_llm = vision_llm
        # Solo mode: pin to a single role, skip transfer loop.
        self._fixed_role = fixed_role
        # Tracks which sessions currently have a turn in flight, mapped to the
        # current_role name at the time the turn started. Used by slash
        # commands (/status, /running, /stop, /new) which need to inspect or
        # control running turns without entering the session lock. Read/written
        # only from the event loop thread (asyncio single-threaded), so no
        # extra locking needed.
        self._busy_sessions: dict[str, str] = {}

    async def handle(self, msg: IncomingMessage, stream: StreamHandle) -> None:
        session = await self.sessions.get_or_create(msg.session_id)
        # Prefer rich content_blocks (multi-modal); fall back to flat text for
        # adapters that haven't been upgraded.
        user_content = msg.content_blocks if msg.content_blocks else msg.text
        # Mark this session as busy BEFORE acquiring the lock so /status and
        # /running (which never take the lock) see an accurate picture even
        # while the turn is waiting on session.lock behind a previous turn.
        self._busy_sessions[session.session_id] = session.current_role
        try:
            await self._handle_locked(session, user_content, stream)
        finally:
            # Clear busy state even if the turn was cancelled via /stop
            # (CancelledError is BaseException, propagates through the lock
            # __aexit__; this finally still runs).
            self._busy_sessions.pop(session.session_id, None)

    async def _handle_locked(
        self,
        session: Session,
        user_content,
        stream: StreamHandle,
    ) -> None:
        async with session.lock:
            # On any failure inside _run_turn, agent.history may have been
            # mutated (user message appended, tool loop partially run). We
            # still need to (1) reset per-turn counters, (2) push something
            # back to the user via the stream, and (3) persist whatever state
            # remains so the next turn doesn't replay the broken transcript.
            final = ""
            heartbeat_task: asyncio.Task | None = None
            heartbeat_stop = asyncio.Event()
            try:
                if self.settings.session.progress_status_enabled:
                    await stream.status("已收到,正在处理...")
                    heartbeat_task = asyncio.create_task(
                        self._progress_heartbeat(stream, heartbeat_stop),
                        name=f"progress-heartbeat-{session.session_id}",
                    )
                final = await self._run_turn(session, user_content, stream)
            except Exception:                                  # noqa: BLE001
                log.exception(
                    "turn failed for session=%s role=%s",
                    session.session_id, session.current_role,
                )
                final = "(系统出错,请稍后再试。)"
            finally:
                heartbeat_stop.set()
                if heartbeat_task is not None:
                    try:
                        await heartbeat_task
                    except Exception:                          # noqa: BLE001
                        log.debug("progress heartbeat ended with error", exc_info=True)
                session.reset_turn_counters()

            # Reply BEFORE post-turn work. _post_turn runs compaction, which
            # may make an LLM round-trip per agent — without this ordering the
            # user would wait for `_run_turn + every agent's summarisation
            # call` before seeing the answer. stream.finish just enqueues a
            # WS frame, so it's effectively instant; compaction can then run
            # while the user reads the reply. We stay inside session.lock so
            # the next turn still waits for compaction (mutates agent.history)
            # and persistence's synchronous snapshot.
            try:
                await stream.finish(final)
            except Exception:                                  # noqa: BLE001
                log.exception(
                    "stream.finish failed for session=%s", session.session_id,
                )
            log.info(
                "turn completed for session=%s role=%s",
                session.session_id, session.current_role,
            )
            try:
                await self._post_turn(session)
            except Exception:                                  # noqa: BLE001
                log.exception(
                    "post_turn failed for session=%s", session.session_id,
                )

    async def _progress_heartbeat(self, stream: StreamHandle, stop: asyncio.Event) -> None:
        delay = max(0.0, float(self.settings.session.progress_status_delay_seconds))
        interval = max(0.2, float(self.settings.session.progress_status_interval_seconds))
        text = (self.settings.session.progress_status_text or "").strip() or "正在处理,请稍候..."

        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except asyncio.TimeoutError:
            pass

        while not stop.is_set():
            try:
                await stream.status(text)
            except Exception:                                  # noqa: BLE001
                log.debug("progress status push failed", exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _post_turn(self, session: Session) -> None:
        for agent in list(session.agents_by_role.values()):
            try:
                await maybe_compact(agent, self.llm)
            except Exception:                                  # noqa: BLE001
                log.exception("compaction failed for role=%s", agent.role.name)
        if self.persistence is not None:
            self.persistence.schedule(session)

    # ---- slash-command support surface -----------------------------------
    # These methods are called by the adapter's slash-command handler. They
    # are designed to be called WITHOUT holding session.lock — they either
    # read lock-free state (busy tracking) or operate on the session metadata
    # files directly. /new refuses if a turn is in flight (caller checks
    # is_busy first); /stop cancels the running task from the adapter side
    # (the adapter owns the inbound worker task) and these methods only
    # provide the read-side introspection.

    def is_busy(self, session_id: str) -> bool:
        """True if a turn is currently in flight for this session."""
        return session_id in self._busy_sessions

    def current_role_for(self, session_id: str) -> str | None:
        """The role name of the in-flight turn, or None if idle. Useful for
        /status replies so the user knows which employee is working."""
        return self._busy_sessions.get(session_id)

    def busy_group_sessions(self) -> list[str]:
        """Session IDs of group chats with an in-flight turn. Used by the
        private-chat /running command. Filters by the WeCom group session_id
        prefix; adapters using a different prefix scheme should override."""
        return [
            sid for sid in self._busy_sessions
            if sid.startswith("wecom-group-")
        ]

    async def reset_session_history(self, session_id: str) -> str:
        """Clear all per-role conversation histories for a session, leaving
        workspace files (inbox/, .chat_team/runs/, .chat_team/llm/,
        notebook.md) untouched. Preserves current_role. Returns the
        current_role so the caller can echo it in the reply.

        MUST be called only when is_busy(session_id) is False — clearing
        histories while a turn is running would race the agent's history
        mutations. The adapter's slash handler checks is_busy() first and
        refuses /new if busy.

        We do NOT acquire session.lock here: if the session is idle the lock
        is uncontended, and the only in-memory state we mutate is
        agents_by_role (clearing it forces _agent_for to rebuild on next
        turn) plus the on-disk session.json (rewritten atomically)."""
        session = await self.sessions.get_or_create(session_id)
        # Drop in-memory agents so the next turn re-materialises them from
        # the (now empty) restored_histories. Without this, a stale Agent
        # holding its old history would survive the reset.
        session.agents_by_role.clear()
        session.restored_histories.clear()
        # Rewrite session.json with empty histories but the same current_role.
        # Go through persistence.flush_now so the atomic-write + schema stays
        # in one place; if persistence is unwired, fall back to write_atomic
        # directly (the snapshot of a cleared agents_by_role is empty
        # histories, which is exactly what we want).
        from .session.persistence import snapshot, write_atomic
        snap = snapshot(session)  # histories == {} after agents_by_role.clear()
        write_atomic(session.cwd, snap, session.state_filename)
        # Cancel any pending debounced flush so it doesn't clobber our reset
        # with a stale snapshot taken before the clear.
        if self.persistence is not None:
            old = self.persistence._pending.pop(session_id, None)
            if old is not None and not old.done():
                old.cancel()
        log.info(
            "session %s history reset (slash /new); current_role=%s preserved",
            session_id, session.current_role,
        )
        return session.current_role

    async def _run_turn(
        self,
        session: Session,
        user_content,
        stream: StreamHandle,
    ) -> str:
        if self._fixed_role:
            return await self._run_turn_solo(session, user_content, stream)
        cap = self.settings.session.per_turn_transfer_cap
        original_content = user_content
        while True:
            agent = self._agent_for(session, session.current_role)
            self._apply_pending_handoff(agent, session)
            # Apply the *current* role's vision strategy to the *original*
            # user content. On transfer, the new role re-evaluates the same
            # raw image blocks under its own strategy.
            transformed = await apply_vision_strategy(
                original_content,
                role=agent.role,
                settings=self.settings,
                llm=self._vision_llm or self.llm,
                cwd=session.cwd,
                session_id=session.session_id,
            )
            try:
                return await agent.handle(transformed, stream)
            except TransferRequested as t:
                session.transfer_count_this_turn += 1
                if session.transfer_count_this_turn >= cap:
                    log.warning(
                        "session %s hit transfer cap %d; forcing %s to answer",
                        session.session_id, cap, session.current_role,
                    )
                    agent.queue_context_note(
                        f"[强制回答] 你已经在本轮触发过 {cap} 次员工交接,"
                        f"现在必须直接给用户答复,不再调用 transfer_to_employee。"
                    )
                    return await agent.handle(
                        f"(系统提示: 交接次数达上限,请你直接回答用户的最近一次提问。)",
                        stream,
                    )
                if not self.roles.has(t.target):
                    log.warning("transfer target %s unknown; staying with %s", t.target, session.current_role)
                    agent.queue_context_note(
                        f"[交接失败] 没有名为 {t.target} 的同事,请直接回答用户。"
                    )
                    return await agent.handle(
                        "(系统提示: 上一次交接失败,请直接回答用户的最近一次提问。)",
                        stream,
                    )
                await stream.status(f"交接给 {self.roles.get(t.target).display_name}")
                session.pending_handoff = PendingHandoff(
                    from_role=session.current_role,
                    to_role=t.target,
                    reason=t.reason,
                    note=t.handoff_note,
                )
                session.current_role = t.target
                # loop again with the new role and the same user_text

    async def _run_turn_solo(
        self,
        session: Session,
        user_content,
        stream: StreamHandle,
    ) -> str:
        session.current_role = self._fixed_role  # type: ignore[assignment]
        agent = self._agent_for(session, self._fixed_role)  # type: ignore[arg-type]
        transformed = await apply_vision_strategy(
            user_content,
            role=agent.role,
            settings=self.settings,
            llm=self._vision_llm or self.llm,
            cwd=session.cwd,
            session_id=session.session_id,
        )
        try:
            return await agent.handle(transformed, stream)
        except TransferRequested:
            log.warning(
                "session %s: transfer attempted in solo mode; ignoring",
                session.session_id,
            )
            agent.queue_context_note("[系统] 当前为独立模式,无法转接给其他员工,请直接回答用户。")
            return await agent.handle(
                "(系统提示: 当前为独立模式,请直接回答用户的提问。)",
                stream,
            )

    def _apply_pending_handoff(self, agent: Agent, session: Session) -> None:
        if not session.pending_handoff:
            return
        h = session.pending_handoff
        if h.to_role != agent.role.name:
            return
        agent.queue_context_note(
            f"[交接备忘] 来自同事 {h.from_role}。原因: {h.reason}\n备忘: {h.note}\n"
            "请基于此备忘和团队记事本继续服务用户;若你也认为应该再次转给别人,请慎重。"
        )
        session.pending_handoff = None

    def _agent_for(self, session: Session, role_name: str) -> Agent:
        if role_name in session.agents_by_role:
            return session.agents_by_role[role_name]
        role = self.roles.get(role_name)
        agent = Agent(
            role=role,
            session=session,
            settings=self.settings,
            llm=self.llm,
            tools=self.tools,
            skills=self.skills,
            vision_llm=self._vision_llm,
        )
        # First time we materialise this role for the session — adopt any
        # history loaded from disk and consume it (one-shot).
        prior = session.restored_histories.pop(role_name, None)
        if prior:
            agent.history = list(prior)
        session.agents_by_role[role_name] = agent
        return agent
