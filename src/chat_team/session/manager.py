"""SessionManager: lookup-or-create per-conversation Session objects.

Two ops layered on top of the basic cache:

* **LRU eviction** — ``max_in_memory_sessions`` caps the in-memory dict.
  When exceeded, the least-recently-used Session is flushed to its own
  ``session.json`` via the injected :class:`PersistenceManager` and then
  dropped. On the user's next message, ``load_state`` + ``restored_histories``
  rebuild it transparently.

* **Lazy file sweep** — first ``get_or_create`` per session (and once every
  ``sweep_interval_hours`` thereafter) walks ``inbox/``, ``.chat_team/runs/``,
  and ``.chat_team/llm/`` and unlinks files older than ``max_age_days``.
  Without this the workspace grows forever.

Both ``get_or_create`` and ``_evict_if_needed`` are async because they
perform disk I/O — JSON load on cold-cache restore, JSON dump on eviction.
``get_or_create`` uses double-checked locking around ``_create_lock`` so
two concurrent inbound messages for the same fresh session_id both end up
sharing one Session object (and one ``session.lock``) instead of one
silently overwriting the other.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..paths import sanitize_session_id
from .notebook import Notebook
from .persistence import load_state, restored_histories
from .session import Session

if TYPE_CHECKING:
    from .persistence import PersistenceManager

log = logging.getLogger(__name__)


def _session_uuid_from_state(state: dict) -> str:
    """Read a persisted conversation UUID, or create one for legacy state."""
    raw = state.get("session_uuid")
    try:
        return str(uuid.UUID(str(raw)))
    except (AttributeError, ValueError, TypeError):
        return str(uuid.uuid4())


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        persistence: "PersistenceManager | None" = None,
        *,
        solo_role: str | None = None,
    ):
        self.settings = settings
        self._solo_role = solo_role
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        # Optional — only used to force a synchronous flush before evicting
        # an LRU entry. SessionManager works without it (the eviction just
        # drops the in-memory copy and the next access reloads from disk;
        # any unflushed delta is lost, so always wire it in production).
        self._persistence = persistence
        self._last_sweep: dict[str, float] = {}
        # Serialises the cold-path create + eviction window across concurrent
        # callers. Cache hits do NOT take this lock — only first-touch for a
        # given session_id and the eviction it may trigger. Cheap creation
        # path → near-zero contention in steady state.
        self._create_lock = asyncio.Lock()

    def workspace_for(self, session_id: str) -> Path:
        return self.settings.workspace_root / sanitize_session_id(session_id)

    async def get_or_create(self, session_id: str) -> Session:
        # Fast path: cache hit. Move-to-end (recency bump) and a possible
        # janitor sweep are the only work. No lock held.
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            session = self._sessions[session_id]
            await self._maybe_sweep(session)
            return session

        # Cold path: two concurrent inbound messages for the same fresh
        # session_id can both miss the membership check above. Without
        # this lock + re-check, both would construct a Session with
        # distinct asyncio.Locks; the second insertion would silently
        # overwrite the first, and any turn already running on the
        # orphaned Session would lose serialisation against the survivor.
        async with self._create_lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                session = self._sessions[session_id]
            else:
                # Disk I/O (mkdir, JSON load) off the event-loop thread.
                session = await asyncio.to_thread(self._build_session, session_id)
                self._sessions[session_id] = session
                await self._evict_if_needed()

        await self._maybe_sweep(session)
        return session

    def known_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    # ---- internals --------------------------------------------------------

    def _build_session(self, session_id: str) -> Session:
        """Sync creation pipeline. Runs in a worker thread via to_thread so
        load_state's JSON parse + restored_histories' second pass don't
        block the event loop. Called only under ``_create_lock`` so two
        concurrent first-touches for the same id can't both run it."""
        cwd = self.workspace_for(session_id)
        cwd.mkdir(parents=True, exist_ok=True)
        meta = cwd / ".chat_team"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "runs").mkdir(parents=True, exist_ok=True)

        notebook = Notebook(meta / "notebook.md", max_bytes=self.settings.notebook.max_bytes)

        if self._solo_role:
            state_filename = f"session-{self._solo_role}.json"
            current_role = self._solo_role
            prior = load_state(cwd, state_filename) or {}
            prior_histories = restored_histories(cwd, state_filename)
        else:
            state_filename = "session.json"
            prior = load_state(cwd) or {}
            current_role = prior.get("current_role") or self.settings.default_role
            prior_histories = restored_histories(cwd)

        return Session(
            session_id=session_id,
            cwd=cwd,
            current_role=current_role,
            notebook=notebook,
            session_uuid=_session_uuid_from_state(prior),
            restored_histories=prior_histories,
            state_filename=state_filename,
        )

    async def _evict_if_needed(self) -> None:
        """Drop LRU entries past the cap. The pre-eviction flush is the
        slow part — it serialises the full history dict and fsyncs — so we
        run it in a worker thread to keep the event loop responsive (otherwise
        every new session past the cap would block heartbeat, the writer
        task, and every other session for the duration of the write)."""
        cap = self.settings.session.max_in_memory_sessions
        if cap <= 0:
            return
        while len(self._sessions) > cap:
            sid, victim = self._sessions.popitem(last=False)
            self._last_sweep.pop(sid, None)
            if self._persistence is not None:
                try:
                    await asyncio.to_thread(self._persistence.flush_now, victim)
                except Exception:                                # noqa: BLE001
                    log.exception("flush-on-evict failed for %s", sid)
            log.info(
                "evicted session %s (in-memory cap %d reached)", sid, cap,
            )

    async def _maybe_sweep(self, session: Session) -> None:
        cfg = self.settings.cleanup
        if cfg.max_age_days <= 0:
            return
        now = time.monotonic()
        last = self._last_sweep.get(session.session_id)
        if last is not None and (now - last) < cfg.sweep_interval_hours * 3600.0:
            return
        self._last_sweep[session.session_id] = now
        cutoff = time.time() - cfg.max_age_days * 86400.0
        # Walk + unlink can do hundreds of stat/unlink syscalls on a long-idle
        # workspace; push it off the event loop.
        await asyncio.to_thread(
            _sweep_session_dirs,
            session.cwd,
            list(cfg.sweep_subdirs),
            cutoff,
            session.session_id,
        )


def _sweep_session_dirs(
    cwd: Path,
    sweep_subdirs: list[str],
    cutoff_ts: float,
    session_id: str,
) -> None:
    for rel in sweep_subdirs:
        target = cwd / rel
        try:
            _unlink_older_than(target, cutoff_ts)
        except Exception:                                        # noqa: BLE001
            log.warning(
                "sweep failed for %s in session %s",
                target, session_id,
                exc_info=True,
            )


def _unlink_older_than(directory: Path, cutoff_ts: float) -> None:
    """Best-effort: unlink files in ``directory`` whose mtime is older than
    ``cutoff_ts`` (epoch seconds). Subdirectories are not recursed into —
    the sweep targets are flat directories by design. Missing directory is
    silently ignored. Per-file errors are swallowed so one bad inode doesn't
    abort the rest of the sweep."""
    if not directory.exists():
        return
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff_ts:
                        os.unlink(entry.path)
                except OSError:
                    continue
    except FileNotFoundError:
        return
