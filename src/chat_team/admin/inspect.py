"""Server + chat_team inspection helpers.

Framework-agnostic — every function returns plain dict / list / str so the
smoke tests can call them without spinning up aiohttp. All disk I/O and
subprocess work runs in a worker thread (``asyncio.to_thread`` from the
route layer) so the admin event loop is never blocked.

Nothing here touches the chat_team runtime objects — the admin process is
deliberately decoupled. Service status comes from ``systemctl`` (the bot is
a systemd unit ``chat-team.service``), disk from ``os.statvfs``, and the
chat_team footprint from ``du``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SERVICE_NAME = "chat-team.service"
DU_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------
# systemctl wrapper (sync — run via asyncio.to_thread)
# --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    """Run ``cmd`` and return ``(rc, stdout, stderr)``.

    Used for ``systemctl`` calls; never raises — a failed/missing systemctl
    returns a non-zero rc and we degrade gracefully.
    """
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    return p.returncode, p.stdout, p.stderr


def _parse_systemctl_show(stdout: str) -> dict[str, str]:
    """Parse ``systemctl show -p X -p Y`` key=value lines."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _format_uptime(active_enter_timestamp: str) -> str:
    """Convert ``ActiveEnterTimestamp=`` (systemctl show) → human ``Xd Yh Zm``.

    Systemd emits this property in one of two shapes depending on version:

      * newer (≥v251): an integer epoch-microseconds string (e.g. ``1753353118000000``);
      * older / default: a human-readable string (e.g. ``Fri 2026-07-24 05:51:58 UTC``).

    We try integer-parse first (covers both the μs shape and a bare epoch-s
    string with a fallback), then fall back to ``datetime.strptime`` for the
    human form. Returns ``"—"`` only when both fail.
    """
    if not active_enter_timestamp:
        return "—"
    ts: float | None = None
    # Case 1: pure integer → assume microseconds (systemd's newer format).
    try:
        if active_enter_timestamp.isdigit():
            raw = int(active_enter_timestamp)
            # Heuristic: μs timestamps since ~2000 are >= 1e15; seconds since
            # epoch since ~2000 are >= 1e9. Either way we want a wall-clock
            # instant, so convert from whichever unit the magnitude suggests.
            if raw >= 1_000_000_000_000_000:        # microseconds
                ts = raw / 1_000_000
            elif raw >= 1_000_000_000:              # seconds
                ts = float(raw)
            # Smaller numbers are not plausible wall-clock instants; leave ts None.
    except ValueError:
        pass
    # Case 2: human-readable systemd date string.
    if ts is None:
        from datetime import datetime, timezone  # local import; rare path
        for fmt in (
            "%a %Y-%m-%d %H:%M:%S %Z",   # "Fri 2026-07-24 05:51:58 UTC"
            "%a %Y-%m-%d %H:%M:%S",      # without TZ
            "%Y-%m-%d %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(active_enter_timestamp, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
                break
            except ValueError:
                continue
    if ts is None:
        return "—"
    delta = max(0.0, time.time() - ts)
    days = int(delta // 86400)
    hours = int((delta % 86400) // 3600)
    mins = int((delta % 3600) // 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _format_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def get_service_status_sync(service: str = SERVICE_NAME) -> dict[str, Any]:
    """Return chat_team service status from ``systemctl``.

    Fields: ``active`` (bool), ``sub_state`` (str, e.g. "running"),
    ``pid`` (int|None), ``uptime`` (str), ``memory_bytes`` (int|None),
    ``source`` ("systemctl"|"ps-fallback"|"unknown").
    """
    rc, out, err = _run(["systemctl", "show", service, "--no-pager",
                         "-p", "ActiveState",
                         "-p", "SubState",
                         "-p", "MainPID",
                         "-p", "ActiveEnterTimestamp",
                         "-p", "MemoryCurrent"])
    if rc == 0:
        props = _parse_systemctl_show(out)
        active_state = props.get("ActiveState", "").lower()
        sub_state = props.get("SubState", "")
        pid_str = props.get("MainPID", "0")
        try:
            pid = int(pid_str) if pid_str else 0
        except ValueError:
            pid = 0
        mem_str = props.get("MemoryCurrent", "")
        memory_bytes: int | None
        if mem_str in ("", "[not set]"):
            memory_bytes = None
        else:
            try:
                memory_bytes = int(mem_str)
            except ValueError:
                memory_bytes = None
        return {
            "active": active_state == "active",
            "sub_state": sub_state or ("running" if active_state == "active" else "unknown"),
            "pid": pid or None,
            "uptime": _format_uptime(props.get("ActiveEnterTimestamp", "")),
            "memory_bytes": memory_bytes,
            "memory_human": _format_bytes(memory_bytes),
            "source": "systemctl",
        }

    # Fall back: systemctl missing (no systemd) or service not found.
    log.info("systemctl show failed (rc=%d, err=%s); trying ps fallback", rc, err.strip()[:200])
    return _ps_fallback_sync(service)


def _ps_fallback_sync(service: str) -> dict[str, Any]:
    """Best-effort: scan ps for ``main.py`` running with --foreground.

    Used when systemctl is unavailable (no systemd) — gives *some* status.
    """
    rc, out, _ = _run(["ps", "-eo", "pid=,args=", "--no-headers"], timeout=10.0)
    if rc != 0:
        return {
            "active": False,
            "sub_state": "unknown",
            "pid": None,
            "uptime": "—",
            "memory_bytes": None,
            "memory_human": "—",
            "source": "unknown",
        }
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, args = line.split(None, 1)
        except ValueError:
            continue
        if "main.py" in args and ("--foreground" in args or "-f" in args.split()):
            pid = int(pid_str)
            # rss from /proc/<pid>/status
            memory_bytes = _proc_rss_bytes(pid)
            return {
                "active": True,
                "sub_state": "running",
                "pid": pid,
                "uptime": _format_uptime(_proc_start_us(pid)),
                "memory_bytes": memory_bytes,
                "memory_human": _format_bytes(memory_bytes),
                "source": "ps-fallback",
            }
    return {
        "active": False,
        "sub_state": "not-running",
        "pid": None,
        "uptime": "—",
        "memory_bytes": None,
        "memory_human": "—",
        "source": "ps-fallback",
    }


def _proc_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[1]) * 1024  # kB → bytes
                        except ValueError:
                            return None
    except OSError:
        return None
    return None


def _proc_start_us(pid: int) -> str:
    """Read /proc/<pid>/stat start time (jiffies) and convert to μs.

    Returns "" if unavailable; ``_format_uptime`` tolerates that.
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as f:
            parts = f.read().split()
        # Field 22 is starttime in clock ticks since boot.
        if len(parts) >= 22:
            try:
                ticks = int(parts[21])
                # ticks → seconds since boot; we need an absolute μs timestamp.
                clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                boot_uptime_s = _boot_time_seconds()
                if boot_uptime_s is not None:
                    start_uptime_s = ticks / clk
                    abs_epoch_s = boot_uptime_s + start_uptime_s
                    return str(int(abs_epoch_s * 1_000_000))
            except (ValueError, KeyError, ZeroDivisionError):
                pass
    except OSError:
        pass
    return ""


def _boot_time_seconds() -> float | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime "):
                    try:
                        return float(line.split()[1])
                    except (ValueError, IndexError):
                        return None
    except OSError:
        return None
    return None


def restart_service_sync(service: str = SERVICE_NAME, wait_seconds: float = 30.0) -> dict[str, Any]:
    """``systemctl restart`` and poll until active (or timeout)."""
    rc, out, err = _run(["systemctl", "restart", service], timeout=45.0)
    if rc != 0:
        return {"ok": False, "error": err.strip() or f"systemctl restart rc={rc}"}
    # Poll for active.
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        status = get_service_status_sync(service)
        if status["active"]:
            return {"ok": True, "status": "restarted", **status}
        time.sleep(0.5)
    return {"ok": False, "error": f"did not return to active within {wait_seconds:.0f}s"}


def reload_service_sync(service: str = SERVICE_NAME) -> dict[str, Any]:
    """``systemctl reload`` (= SIGHUP → hot-reload config.yaml/team.md/roles)."""
    rc, out, err = _run(["systemctl", "reload", service], timeout=15.0)
    if rc != 0:
        return {"ok": False, "error": err.strip() or f"systemctl reload rc={rc}"}
    return {"ok": True, "status": "reloaded"}


# --------------------------------------------------------------------------
# Disk usage
# --------------------------------------------------------------------------

@dataclass
class DiskUsage:
    partition_total: int
    partition_used: int
    partition_free: int
    partition_use_percent: float
    chat_team_total: int
    chat_team_total_human: str
    subdirs: list[dict[str, Any]]  # [{name, size_bytes, size_human}]
    sessions_top: list[dict[str, Any]]  # [{sid, size_bytes, size_human, mtime}]
    truncated: bool = False


def _path_size_bytes(path: Path) -> int:
    """Best-effort recursive size (bytes). On any error, returns 0."""
    try:
        if path.is_file() or path.is_symlink():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        if path.is_dir():
            total = 0
            for entry in path.rglob("*"):
                try:
                    if entry.is_file() and not entry.is_symlink():
                        total += entry.stat().st_size
                except OSError:
                    continue
            return total
    except OSError:
        pass
    return 0


def _du_bytes(path: Path) -> int:
    """Prefer ``du -sb`` for speed; fall back to a Python walk."""
    if not path.exists():
        return 0
    rc, out, _ = _run(["du", "-sb", str(path)], timeout=DU_TIMEOUT_S)
    if rc == 0:
        # Output: "<bytes>\t<path>"
        first = out.split(None, 1)[0] if out else ""
        try:
            return int(first)
        except ValueError:
            pass
    return _path_size_bytes(path)


def _du_subdir_sync(home: Path, top_n: int = 20) -> DiskUsage:
    """Synchronous disk-usage scan; runs in worker thread."""
    # Partition-level via statvfs on the chat_team home.
    total = used = free = 0
    use_pct = 0.0
    try:
        sv = os.statvfs(home)
        total = sv.f_blocks * sv.f_frsize
        free = sv.f_bavail * sv.f_frsize
        used = total - sv.f_bavail * sv.f_frsize  # rough (excludes reserved root)
        if total > 0:
            use_pct = round((used / total) * 100, 1)
    except OSError as e:
        log.warning("statvfs(%s) failed: %r", home, e)

    # Top-level subdir sizes under chat_team home.
    subdir_names = ["logs", "workspaces", "state", "roles", "skills", "admin"]
    subdirs: list[dict[str, Any]] = []
    for name in subdir_names:
        p = home / name
        if not p.exists():
            continue
        size = _du_bytes(p)
        subdirs.append({"name": name, "size_bytes": size, "size_human": _format_bytes(size)})

    chat_team_total = sum(s["size_bytes"] for s in subdirs)
    # Include root-level files (config.yaml, team.md, .env) so the total is honest.
    for entry in home.iterdir():
        if entry.is_file():
            try:
                chat_team_total += entry.stat().st_size
            except OSError:
                pass

    # Sessions (workspaces/<sid>) — top-N by size.
    sessions_top: list[dict[str, Any]] = []
    ws = home / "workspaces"
    truncated = False
    if ws.is_dir():
        entries: list[tuple[str, int, float]] = []
        for entry in ws.iterdir():
            if not entry.is_dir():
                continue
            size = _du_bytes(entry)
            mtime = 0.0
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                pass
            entries.append((entry.name, size, mtime))
        entries.sort(key=lambda t: t[1], reverse=True)
        if len(entries) > top_n:
            entries = entries[:top_n]
            truncated = True
        for name, size, mtime in entries:
            sessions_top.append({
                "sid": name,
                "size_bytes": size,
                "size_human": _format_bytes(size),
                "mtime": int(mtime) if mtime else 0,
                "mtime_human": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "—",
            })

    return DiskUsage(
        partition_total=total,
        partition_used=used,
        partition_free=free,
        partition_use_percent=use_pct,
        chat_team_total=chat_team_total,
        chat_team_total_human=_format_bytes(chat_team_total),
        subdirs=subdirs,
        sessions_top=sessions_top,
        truncated=truncated,
    )


# --------------------------------------------------------------------------
# Log tail
# --------------------------------------------------------------------------

def tail_log_sync(path: Path, lines: int = 200) -> str:
    """Return the last ``lines`` lines of ``path`` as plain text.

    Prefers ``tail -n`` for speed; falls back to a Python read. Missing
    file → sentinel string the dashboard renders verbatim.
    """
    if not path.exists():
        return f"(no log file at {path})"
    rc, out, _ = _run(["tail", "-n", str(lines), str(path)], timeout=5.0)
    if rc == 0:
        return out
    # Fallback: read whole file, slice last N lines.
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
        return "".join(data.splitlines(keepends=True)[-lines:])
    except OSError as e:
        return f"(failed to read {path}: {e!r})"


def tail_journal_sync(unit: str, lines: int = 200) -> str:
    """Tail ``journalctl -u <unit>`` for systemd-managed services.

    Falls back to ``~/.chat_team/logs/chat_team.log`` if journalctl is
    missing or empty (e.g. bare background-daemon mode).
    """
    rc, out, _ = _run(["journalctl", "-u", unit, "-n", str(lines), "--no-pager"], timeout=10.0)
    if rc == 0 and out.strip():
        return out
    # Fallback to the on-disk rotating log.
    home = Path(os.environ.get("CHAT_TEAM_HOME", str(Path.home() / ".chat_team")))
    return tail_log_sync(home / "logs" / "chat_team.log", lines)


# --------------------------------------------------------------------------
# Caching wrapper (avoid hammering du on a refresh-spamming user)
# --------------------------------------------------------------------------

class CachedDiskInspector:
    """Wraps ``_du_subdir_sync`` with a TTL cache so repeated calls inside
    ``cache_ttl`` return the same result without re-walking the filesystem.
    """

    def __init__(self, home: Path, cache_ttl: float = 30.0):
        self._home = home
        self._ttl = cache_ttl
        self._cached: DiskUsage | None = None
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    async def inspect(self) -> DiskUsage:
        async with self._lock:
            now = time.time()
            if self._cached is not None and (now - self._cached_at) < self._ttl:
                return self._cached
            result = await asyncio.to_thread(_du_subdir_sync, self._home)
            self._cached = result
            self._cached_at = now
            return result


# --------------------------------------------------------------------------
# Async wrappers (call these from aiohttp routes)
# --------------------------------------------------------------------------

async def get_service_status(service: str = SERVICE_NAME) -> dict[str, Any]:
    return await asyncio.to_thread(get_service_status_sync, service)


async def restart_service(service: str = SERVICE_NAME, wait_seconds: float = 30.0) -> dict[str, Any]:
    return await asyncio.to_thread(restart_service_sync, service, wait_seconds)


async def reload_service(service: str = SERVICE_NAME) -> dict[str, Any]:
    return await asyncio.to_thread(reload_service_sync, service)


async def tail_journal(unit: str, lines: int = 200) -> str:
    return await asyncio.to_thread(tail_journal_sync, unit, lines)


async def tail_log(path: Path, lines: int = 200) -> str:
    return await asyncio.to_thread(tail_log_sync, path, lines)
