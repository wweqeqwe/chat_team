"""Fail-closed bridge from the local handoff sidecar into service_advisor."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .adapters.base import ChatType, ContentBlock, IncomingMessage


log = logging.getLogger(__name__)
_GROUP_BIND_RE = re.compile(
    r"(?:绑定|群绑定|绑定车辆群|车辆群(?:搜索)?键)\s*[:：]?\s*([ALQ])\s*"
    r"([\u4e00-\u9fff][A-Z][A-Z0-9]{5,6})",
    re.IGNORECASE,
)
_MANUAL_ORDER_RE = re.compile(r"委托书号\s*[:：]\s*([A-Za-z0-9_-]{3,64})", re.IGNORECASE)
_DATA_URL_RE = re.compile(r"^data:(image/(?:jpeg|png));base64,([A-Za-z0-9+/=\r\n]+)$")


class AdvisorBridgeError(RuntimeError):
    pass


class AdvisorSidecarClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8791",
        token_file: str | Path = "/etc/gongchuang-advisor-handoff.env",
    ):
        self.base_url = base_url.rstrip("/")
        self.token_file = Path(token_file)

    def _token(self) -> str:
        for line in self.token_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("WECOM_SMARTSHEET_RELAY_TOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
        raise AdvisorBridgeError("服务顾问旁路密钥不存在")

    def _post_sync(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except Exception:  # noqa: BLE001
                detail = None
            raise AdvisorBridgeError(detail or f"服务顾问旁路请求失败：HTTP {exc.code}") from exc
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise AdvisorBridgeError("服务顾问旁路暂时不可用") from exc
        if not isinstance(result, dict):
            raise AdvisorBridgeError("服务顾问旁路返回格式不正确")
        return result

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post_sync, path, payload)

    async def lease(self, worker_id: str) -> dict[str, Any] | None:
        return (
            await self.post(
                "/internal/advisor-handoffs/lease",
                {"workerId": worker_id, "leaseSeconds": 180},
            )
        ).get("job")

    async def start(self, job: dict[str, Any], worker_id: str) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-handoffs/start",
            {
                "id": job["id"],
                "leaseToken": job["leaseToken"],
                "workerId": worker_id,
            },
        )

    async def complete(
        self,
        job: dict[str, Any],
        worker_id: str,
        status: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-handoffs/complete",
            {
                "id": job["id"],
                "leaseToken": job["leaseToken"],
                "workerId": worker_id,
                "status": status,
                "result": result,
            },
        )

    async def resolve_group(
        self,
        group_search_key: str,
        *,
        advisor_user_id: str = "",
        advisor_name: str = "",
    ) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-groups/resolve",
            {
                "groupSearchKey": group_search_key,
                "advisorUserId": advisor_user_id,
                "advisorName": advisor_name,
                "todayOnly": True,
            },
        )

    async def broadcast_fallback(
        self,
        job: dict[str, Any],
        worker_id: str,
    ) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-handoffs/fallback-broadcast",
            {
                "id": job["id"],
                "leaseToken": job["leaseToken"],
                "workerId": worker_id,
            },
        )

    async def bind_group(self, msg: IncomingMessage, group_search_key: str) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-groups/bind",
            {
                "groupSearchKey": group_search_key,
                "chatId": msg.chat_id,
                "botId": msg.bot_id,
                "sourceMessageId": msg.msg_id,
                "advisorUserId": msg.user_id,
                "advisorName": msg.user_name,
            },
        )

    async def claim_manual(
        self,
        msg: IncomingMessage,
        *,
        work_order_key: str,
        group_search_key: str,
    ) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-handoffs/manual-key-claim",
            {
                "workOrderKey": work_order_key,
                "groupSearchKey": group_search_key,
                "advisorUserId": msg.user_id,
                "advisorName": msg.user_name,
            },
        )

    async def complete_manual(
        self,
        *,
        work_order_key: str,
        claim_token: str,
        status: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.post(
            "/internal/advisor-handoffs/manual-complete",
            {
                "workOrderKey": work_order_key,
                "claimToken": claim_token,
                "status": status,
                "result": result,
            },
        )


class AdvisorProactiveStream:
    """Dispatcher stream that sends only the final answer as a proactive message."""

    def __init__(self, adapter: Any, chat_id: str):
        self.adapter = adapter
        self.chat_id = chat_id
        self.content = ""
        self.sent = False
        self.error = ""

    async def push(self, chunk: str, *, append: bool = True) -> None:
        if not chunk.strip():
            return
        self.content = self.content + chunk if append else chunk

    async def status(self, note: str) -> bool:
        return True

    async def finish(self, final_text: str) -> None:
        try:
            await self.adapter.send_proactive_markdown(
                self.chat_id,
                final_text or "(空回复)",
            )
            self.sent = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            raise

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:
        await self.adapter.send_proactive_media(
            self.chat_id,
            path,
            kind="image",
            filename=filename,
        )

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:
        await self.adapter.send_proactive_media(
            self.chat_id,
            path,
            kind="file",
            filename=filename,
        )


def binding_key_from_message(msg: IncomingMessage) -> str | None:
    if msg.chat_type != ChatType.GROUP or not msg.chat_id or not msg.msg_id:
        return None
    match = _GROUP_BIND_RE.search(msg.text or "")
    if not match:
        return None
    return (match.group(1) + match.group(2)).upper().replace(" ", "")


def manual_order_from_message(msg: IncomingMessage) -> tuple[str, str] | None:
    """Return the idempotency key and explicit vehicle-group key for old mode."""
    if msg.chat_type != ChatType.GROUP or "开服务项目单" not in (msg.text or ""):
        return None
    work_order = _MANUAL_ORDER_RE.search(msg.text or "")
    group_key = binding_key_from_message(msg)
    if not work_order or not group_key:
        return None
    return work_order.group(1), group_key


def _safe_job_name(job_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(job_id or ""))[:64]
    if not safe:
        raise AdvisorBridgeError("服务顾问任务编号无效")
    return safe


def build_job_blocks(
    job: dict[str, Any],
    workspace: Path,
) -> tuple[str, list[ContentBlock], list[Path]]:
    payload = job.get("payload") or {}
    images = payload.get("images") or []
    expected_kinds = ["vehicle_watermark", "electronic_work_order", "nameplate"]
    # The current electronic work-order flow sends structured step-one data
    # directly and does not require the retired three-image envelope.  Keep
    # accepting all three legacy images during migration, but fail closed on
    # a partial or reordered legacy envelope.
    if images and len(images) != 3:
        raise AdvisorBridgeError("旧版服务顾问图片必须保持三张")
    if images and [image.get("kind") for image in images] != expected_kinds:
        raise AdvisorBridgeError("服务顾问三张图片顺序不正确")

    inbox = workspace / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    job_name = _safe_job_name(job.get("id") or "")
    labels = ["车辆水印图", "电子委托书图", "铭牌图"]
    blocks: list[ContentBlock] = []
    paths: list[Path] = []
    projects = payload.get("serviceProjects") or []
    project_lines = "\n".join(
        f"- {str(item.get('name') or '').strip()}"
        for item in projects
        if str(item.get("name") or "").strip()
    ) or "- 未提供"
    vehicle = payload.get("vehicle") or {}
    customer = payload.get("customer") or {}
    tags = payload.get("tags") or {}
    tag_lines = []
    for label, name in (
        ("进厂原因", "reason"),
        ("客户渠道", "channel"),
        ("路况", "road"),
        ("用车习惯", "habit"),
    ):
        values = [str(value).strip() for value in (tags.get(name) or []) if str(value).strip()]
        if values:
            tag_lines.append(f"{label}：{'、'.join(values)}")
    tag_text = "\n".join(tag_lines)
    if tag_text:
        tag_text += "\n"
    summary = (
        "[电子委托书直达任务]\n"
        f"任务号：{job_name}\n"
        f"门店：{payload.get('storeName') or ''}\n"
        f"车牌：{payload.get('plate') or ''}\n"
        f"车型：{vehicle.get('model') or ''}\n"
        f"年份：{vehicle.get('year') or ''}\n"
        f"VIN：{vehicle.get('vin') or ''}\n"
        f"里程：{vehicle.get('mileage') or ''}\n"
        f"客户：{customer.get('name') or ''}\n"
        f"电话：{customer.get('phone') or ''}\n"
        f"服务顾问：{payload.get('advisorName') or ''}\n"
        f"{tag_text}"
        f"服务项目：\n{project_lines}\n"
        "处理要求：按现有服务顾问流程处理服务项目单；不要处理车主主诉或会员套餐。"
    )
    blocks.append({"type": "text", "text": summary})

    for index, (image, expected_kind, label) in enumerate(
        zip(images, expected_kinds, labels),
        start=1,
    ):
        match = _DATA_URL_RE.fullmatch(str(image.get("imageDataUrl") or ""))
        if not match or image.get("kind") != expected_kind:
            raise AdvisorBridgeError(f"{label}编码无效")
        raw = base64.b64decode(match.group(2), validate=True)
        if not 32 <= len(raw) <= 2_000_000:
            raise AdvisorBridgeError(f"{label}大小不正确")
        suffix = ".jpg" if match.group(1) == "image/jpeg" else ".png"
        path = inbox / f"advisor-{job_name}-{index}{suffix}"
        path.write_bytes(raw)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        blocks.append({"type": "text", "text": f"{index}. {label}"})
        blocks.append({"type": "image", "path": f"./inbox/{path.name}"})
        paths.append(path)
    return summary, blocks, paths


class AdvisorDirectBridge:
    def __init__(self, adapter: Any, dispatcher: Any, client: AdvisorSidecarClient | None = None):
        self.adapter = adapter
        self.dispatcher = dispatcher
        self.client = client or AdvisorSidecarClient()
        self.worker_id = f"cloud-service-advisor-{socket.gethostname()}"

    async def observe_message(self, msg: IncomingMessage) -> None:
        key = binding_key_from_message(msg)
        if key:
            await self.client.bind_group(msg, key)
            log.info("advisor vehicle group binding recorded for key=%s", key)

    async def handle_incoming(self, msg: IncomingMessage, stream: Any) -> None:
        """Preserve old @ mode while sharing one dedup lock with direct mode."""
        key = binding_key_from_message(msg)
        if key:
            try:
                await self.client.bind_group(msg, key)
                log.info(
                    "advisor vehicle group binding recorded for key=%s advisor=%s",
                    key,
                    msg.user_id,
                )
            except Exception:
                # The bridge is an additive feature.  A sidecar outage must never
                # interrupt the original group @ workflow.
                log.exception("advisor group binding failed open for key=%s", key)
        manual = manual_order_from_message(msg)
        if manual is None:
            await self.dispatcher.handle(msg, stream)
            return

        work_order_key, group_search_key = manual
        try:
            claim = await self.client.claim_manual(
                msg,
                work_order_key=work_order_key,
                group_search_key=group_search_key,
            )
        except Exception:
            log.exception(
                "advisor dedup claim failed open for work_order=%s", work_order_key
            )
            await self.dispatcher.handle(msg, stream)
            return
        if not claim.get("accepted"):
            await stream.finish(
                f"委托书 {work_order_key} 已由"
                f"{'\u7535\u5b50\u59d4\u6258\u4e66\u76f4\u8fbe\u5165\u53e3' if claim.get('source') == 'direct' else '\u7fa4\u5185\u5165\u53e3'}"
                "接收，本次不再重复处理。"
            )
            return

        token = str(claim.get("claimToken") or "")
        try:
            await self.dispatcher.handle(msg, stream)
        except Exception as exc:
            try:
                await self.client.complete_manual(
                    work_order_key=work_order_key,
                    claim_token=token,
                    status="failed",
                    result={
                        "message": f"群内旧模式处理失败：{str(exc)[:300]}",
                        "route": "manual_group",
                        "groupSearchKey": group_search_key,
                        "robotConfigChanged": False,
                        "wecomMessageSent": False,
                    },
                )
            except Exception:
                log.exception(
                    "advisor failure completion report failed for work_order=%s",
                    work_order_key,
                )
            raise
        try:
            await self.client.complete_manual(
                work_order_key=work_order_key,
                claim_token=token,
                status="succeeded",
                result={
                    "message": "群内旧模式处理完成",
                    "route": "manual_group",
                    "groupSearchKey": group_search_key,
                    "robotConfigChanged": False,
                    "wecomMessageSent": True,
                },
            )
        except Exception:
            # Reporting does not change the already-completed original action.
            log.exception(
                "advisor success completion report failed for work_order=%s",
                work_order_key,
            )

    async def run_once(self) -> bool:
        job = await self.client.lease(self.worker_id)
        if not job:
            return False
        payload = job.get("payload") or {}
        key = str(payload.get("groupSearchKey") or "")
        route = await self.client.resolve_group(
            key,
            advisor_user_id=str(payload.get("submittedBy") or ""),
            advisor_name=str(payload.get("submittedByName") or payload.get("advisorName") or ""),
        )

        if job.get("simulation"):
            await self.client.start(job, self.worker_id)
            await self.client.complete(
                job,
                self.worker_id,
                "simulated",
                {
                    "message": "模拟完成；未发送企业微信消息，未调用 F6",
                    "route": route.get("route") or "simulation",
                    "groupSearchKey": key,
                    "imageCount": len(payload.get("images") or []),
                    "serviceProjectCount": len(payload.get("serviceProjects") or []),
                    "robotConfigChanged": False,
                    "wecomMessageSent": False,
                },
            )
            return True

        if not route.get("found") or not route.get("eligibleForProactiveSend"):
            await self.client.start(job, self.worker_id)
            try:
                fallback = await self.client.broadcast_fallback(job, self.worker_id)
            except Exception as exc:  # noqa: BLE001
                # A network timeout cannot prove the webhook did not accept
                # the notice.  Stop here and require a human check.
                await self.client.complete(
                    job,
                    self.worker_id,
                    "uncertain",
                    {
                        "message": f"门店总群播报结果不确定，需要人工核对：{str(exc)[:300]}",
                        "route": "store_broadcast_needs_review",
                        "groupSearchKey": key,
                        "imageCount": len(payload.get("images") or []),
                        "serviceProjectCount": len(payload.get("serviceProjects") or []),
                        "robotConfigChanged": False,
                        "wecomMessageSent": True,
                    },
                )
                return True
            sent = bool(fallback.get("sent"))
            await self.client.complete(
                job,
                self.worker_id,
                "succeeded" if sent else "uncertain",
                {
                    "message": "没有已绑定车辆群，已播报到对应门店总群"
                    if sent
                    else "门店总群未确认发送，需要人工核对",
                    "route": "store_broadcast",
                    "groupSearchKey": key,
                    "imageCount": len(payload.get("images") or []),
                    "serviceProjectCount": len(payload.get("serviceProjects") or []),
                    "robotConfigChanged": False,
                    "wecomMessageSent": sent,
                },
            )
            return True

        await self.client.start(job, self.worker_id)
        chat_id = str(route["chatId"])
        session_id = f"wecom-group-{chat_id}"
        workspace = self.dispatcher.sessions.workspace_for(session_id)
        summary, blocks, paths = build_job_blocks(job, workspace)
        stream = AdvisorProactiveStream(self.adapter, chat_id)
        wecom_send_attempted = False
        try:
            # From this point onward a timeout is ambiguous: WeCom may have
            # accepted a message even if its acknowledgement was lost.  The
            # job therefore fails closed and must never be retried blindly.
            wecom_send_attempted = True
            await self.adapter.send_proactive_markdown(chat_id, summary)
            for path in paths:
                await self.adapter.send_proactive_media(chat_id, path, kind="image")
            incoming = IncomingMessage(
                session_id=session_id,
                chat_type=ChatType.GROUP,
                user_id=str(payload.get("submittedBy") or "advisor-direct"),
                user_name=str(payload.get("submittedByName") or ""),
                text=summary,
                msg_id=str(job["id"]),
                bot_id=str(route["botId"]),
                chat_id=chat_id,
                raw={
                    "source": "advisor_handoff",
                    "workOrderKey": payload.get("workOrderKey"),
                },
                content_blocks=blocks,
            )
            await self.dispatcher.handle(incoming, stream)
        except Exception as exc:  # noqa: BLE001
            await self.client.complete(
                job,
                self.worker_id,
                "uncertain",
                {
                    "message": f"直达处理结果不确定，需要人工核对：{str(exc)[:300]}",
                    "route": "vehicle_group_needs_review",
                    "groupSearchKey": key,
                    "imageCount": len(paths),
                    "serviceProjectCount": len(payload.get("serviceProjects") or []),
                    "robotConfigChanged": False,
                    "wecomMessageSent": wecom_send_attempted or stream.sent,
                },
            )
            return True

        status = "succeeded" if stream.sent else "uncertain"
        await self.client.complete(
            job,
            self.worker_id,
            status,
            {
                "message": "服务顾问直达任务处理完成" if stream.sent else "未确认群内回复，需要人工核对",
                "route": route.get("route") or "advisor_matching_vehicle_group",
                "groupSearchKey": key,
                "imageCount": len(paths),
                "serviceProjectCount": len(payload.get("serviceProjects") or []),
                "robotConfigChanged": False,
                "wecomMessageSent": stream.sent,
            },
        )
        return True

    async def run_forever(self) -> None:
        while True:
            delay = 0.5
            try:
                handled = await self.run_once()
                if not handled:
                    delay = 2.0
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("advisor direct bridge poll failed")
                delay = 5.0
            await asyncio.sleep(delay)
