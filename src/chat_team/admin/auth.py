"""Auth for the chat_team admin panel.

* ``UserStore`` — reads/writes ``~/.chat_team/admin/users.json``; passwords
  are hashed with PBKDF2-HMAC-SHA256 (stdlib ``hashlib.pbkdf2_hmac``,
  600000 iterations — meets OWASP 2023). No pip dependency.
* ``SessionStore`` — in-memory ``{sid: {username, expires_at, csrf_token, ip}}``.
  Session id is 32 bytes from ``secrets.token_urlsafe``. ``validate_cookie``
  rotates expiry on each use (sliding window) so an active user never gets
  logged out mid-session.
* ``LoginRateLimiter`` — sliding 5-minute window per IP; failed logins are
  counted, successful logins are not (so the right password never trips the
  limit). The 6th failure inside the window returns True (rate-limited).
* ``AuditLogger`` — appends ``[ISO ts] event=... user=... ip=... ua=...``
  to ``~/.chat_team/logs/admin.log`` (configurable). Best-effort: a write
  failure logs a WARNING and the call still returns normally (an audit-log
  hiccup must never take down restart/reload).

These primitives are deliberately framework-agnostic so the smoke tests can
exercise them directly without spinning up aiohttp.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# OWASP 2023 minimum for PBKDF2-HMAC-SHA256.
PBKDF2_ITERATIONS = 600_000
PBKDF2_HASH_NAME = "sha256"
PBKDF2_DK_LEN = 32  # 256 bits

PBKDF2_ALGO_TAG = "pbkdf2_sha256"


# --------------------------------------------------------------------------
# UserStore
# --------------------------------------------------------------------------

@dataclass
class User:
    username: str
    algo: str = PBKDF2_ALGO_TAG
    iterations: int = PBKDF2_ITERATIONS
    salt: str = ""          # hex
    hash: str = ""          # hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "algo": self.algo,
            "iterations": self.iterations,
            "salt": self.salt,
            "hash": self.hash,
        }


class UserStore:
    """JSON-backed user list at ``users_path``.

    The on-disk shape is::

        {"users": [{username, algo, iterations, salt, hash}, ...]}

    Writes are atomic (temp + os.replace) so a crash mid-write never leaves
    a half-written users.json. The store is re-read from disk on every
    lookup — adding a user via the CLI takes effect on the next request
    without restarting the admin process.
    """

    def __init__(self, users_path: Path):
        self.path = users_path

    def ensure_file(self) -> None:
        """Create the parent dir + an empty users.json if missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[User]:
        self.ensure_file()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("users.json unreadable (%r); treating as empty", e)
            return []
        users_raw = data.get("users") if isinstance(data, dict) else None
        if not isinstance(users_raw, list):
            return []
        out: list[User] = []
        for entry in users_raw:
            if not isinstance(entry, dict):
                continue
            username = entry.get("username")
            if not isinstance(username, str) or not username:
                continue
            out.append(User(
                username=username,
                algo=str(entry.get("algo", PBKDF2_ALGO_TAG)),
                iterations=int(entry.get("iterations", PBKDF2_ITERATIONS) or PBKDF2_ITERATIONS),
                salt=str(entry.get("salt", "")),
                hash=str(entry.get("hash", "")),
            ))
        return out

    def _write(self, users: list[User]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"users": [u.to_dict() for u in users]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def list_users(self) -> list[str]:
        return [u.username for u in self._read()]

    def get(self, username: str) -> User | None:
        for u in self._read():
            if u.username == username:
                return u
        return None

    def add_or_update(self, user: User) -> None:
        users = self._read()
        for i, u in enumerate(users):
            if u.username == user.username:
                users[i] = user
                self._write(users)
                return
        users.append(user)
        self._write(users)

    @staticmethod
    def hash_password(password: str) -> tuple[str, int, str]:
        """Return ``(salt_hex, iterations, hash_hex)`` for ``password``."""
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac(PBKDF2_HASH_NAME, password.encode("utf-8"), salt, PBKDF2_ITERATIONS, PBKDF2_DK_LEN)
        return salt.hex(), PBKDF2_ITERATIONS, dk.hex()

    @staticmethod
    def verify(password: str, user: "User | None") -> bool:
        """Constant-time compare of ``password`` against ``user``'s hash.

        Tolerates ``user=None`` (e.g. when the username doesn't exist) so the
        caller can write ``UserStore.verify(pw, store.get(name))`` without a
        separate None check. Returns False for unknown users — and still
        runs one pbkdf2 derivation against a dummy hash so the response time
        is indistinguishable from a real wrong-password attempt (mitigates
        username-enumeration via timing).
        """
        if user is None or not user.salt or not user.hash:
            # Burn time on a dummy pbkdf2 so the unknown-user path isn't
            # observably faster than the wrong-password path.
            hashlib.pbkdf2_hmac(
                PBKDF2_HASH_NAME, password.encode("utf-8"),
                b"\x00" * 16, PBKDF2_ITERATIONS, PBKDF2_DK_LEN,
            )
            return False
        try:
            salt = bytes.fromhex(user.salt)
            expected = bytes.fromhex(user.hash)
        except ValueError:
            return False
        iterations = user.iterations or PBKDF2_ITERATIONS
        actual = hashlib.pbkdf2_hmac(PBKDF2_HASH_NAME, password.encode("utf-8"), salt, iterations, PBKDF2_DK_LEN)
        return hmac.compare_digest(actual, expected)


# --------------------------------------------------------------------------
# SessionStore
# --------------------------------------------------------------------------

@dataclass
class SessionInfo:
    username: str
    expires_at: float          # epoch seconds (absolute)
    csrf_token: str
    ip: str = ""
    created_at: float = 0.0


class SessionStore:
    """In-memory session store. Single-process; thread-safe via a lock.

    A session is created on successful login, looked up on every request via
    ``validate_cookie``, and destroyed by ``/logout`` or when it expires.
    ``validate_cookie`` re-arms the sliding expiry on each successful hit so
    an active user is never logged out mid-session; only idle users fall off.
    """

    def __init__(self, idle_seconds: float):
        self._idle_seconds = idle_seconds
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    def create(self, username: str, ip: str = "") -> tuple[str, str]:
        """Return ``(session_id, csrf_token)``."""
        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        now = time.time()
        with self._lock:
            self._sessions[sid] = SessionInfo(
                username=username,
                expires_at=now + self._idle_seconds,
                csrf_token=csrf,
                ip=ip,
                created_at=now,
            )
        return sid, csrf

    def validate(self, sid: str) -> SessionInfo | None:
        """Return the session if valid (and re-arm its sliding expiry)."""
        if not sid:
            return None
        now = time.time()
        with self._lock:
            info = self._sessions.get(sid)
            if info is None:
                return None
            if info.expires_at <= now:
                self._sessions.pop(sid, None)
                return None
            info.expires_at = now + self._idle_seconds  # sliding window
            return info

    def destroy(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            stale = [s for s, info in self._sessions.items() if info.expires_at <= now]
            for s in stale:
                self._sessions.pop(s, None)
        return len(stale)


# --------------------------------------------------------------------------
# LoginRateLimiter
# --------------------------------------------------------------------------

@dataclass
class _Window:
    failures: list[float] = field(default_factory=list)  # epoch timestamps of failed logins


class LoginRateLimiter:
    """Sliding 5-minute window per IP. Counts *failed* logins only.

    A successful login does NOT count: the right password must never trip
    the limit. ``record_failure`` returns True when the IP has hit the cap
    and should now be rejected (429) before even checking credentials.
    """

    def __init__(self, max_failures: int, window_seconds: float = 300.0):
        self._max = max_failures
        self._window_s = window_seconds
        self._buckets: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def is_rate_limited(self, ip: str) -> bool:
        """True if this IP is currently over the failed-login cap."""
        now = time.time()
        with self._lock:
            w = self._buckets.get(ip)
            if w is None:
                return False
            cutoff = now - self._window_s
            w.failures = [t for t in w.failures if t > cutoff]
            return len(w.failures) >= self._max

    def record_failure(self, ip: str) -> bool:
        """Record a failed login; return True if the IP is now over the cap."""
        now = time.time()
        with self._lock:
            w = self._buckets.setdefault(ip, _Window())
            cutoff = now - self._window_s
            w.failures = [t for t in w.failures if t > cutoff]
            w.failures.append(now)
            return len(w.failures) >= self._max


# --------------------------------------------------------------------------
# AuditLogger
# --------------------------------------------------------------------------

class AuditLogger:
    """Append one line per auditable event to a log file.

    Format::

        [2025-07-24T12:34:56Z] event=login_success user=alice ip=1.2.3.4 ua=Mozilla/5.0

    Backed by a :class:`logging.handlers.RotatingFileHandler` so the file
    can't grow without bound. The handler is constructed once per
    ``AuditLogger`` instance (i.e. once per admin-process start) with
    ``max_bytes``/``backup_count`` from :class:`Settings.admin`. Writes are
    best-effort: a write failure logs a WARNING and the call returns
    normally — the panel must keep working even if the audit log is on a
    read-only mount. The underlying handler is thread-safe (it's a stdlib
    logging handler), so concurrent audit calls from different aiohttp
    routes are safe.

    The :class:`RotatingFileHandler` swaps ``admin.log`` →
    ``admin.log.1`` → ... → ``admin.log.<backup_count>`` in-place when the
    file crosses ``max_bytes``. The previous append-only behaviour is
    preserved for the in-process caller (``log()`` accepts the same kwargs
    and writes one line per call), but the file on disk is now bounded.
    """

    def __init__(
        self,
        log_path: Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.path = log_path
        # A private logger + handler so audit lines don't bleed into the
        # root logger (which would double-write them to chat_team.log via
        # configure_logging's StreamHandler). The logger name is unique
        # enough to avoid accidental collisions; `propagate=False` is the
        # real guarantee.
        self._logger = logging.getLogger(f"chat_team.audit.{log_path.name}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        # Strip any handlers from a previous instance (e.g. test re-init on
        # the same path) so we don't accumulate duplicates.
        for h in list(self._logger.handlers):
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
            self._logger.removeHandler(h)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max(1, int(max_bytes)),
                backupCount=max(0, int(backup_count)),
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._handler = handler
        except OSError as e:
            # Disk full / read-only mount / permission denied — fall back to
            # the previous best-effort semantics: keep the path for
            # tail_log in the panel, but writes will WARN-and-drop.
            log.warning(
                "audit log handler init failed for %s (%r); audit events "
                "will be dropped", log_path, e,
            )
            self._handler = None

    def log(self, event: str, *, user: str = "", ip: str = "", ua: str = "", extra: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [f"event={event}"]
        if user:
            parts.append(f"user={user}")
        if ip:
            parts.append(f"ip={ip}")
        if ua:
            # Compact UA — strip whitespace so the line stays single-line.
            parts.append(f"ua={' '.join(ua.split()[:6])}")
        if extra:
            parts.append(extra)
        line = f"[{ts}] {' '.join(parts)}"
        if self._handler is None:
            log.warning("audit log write skipped (handler init failed); event=%s lost", event)
            return
        try:
            # RotatingFileHandler emits with a trailing newline via the
            # %(message)s formatter, so we don't append "\n" ourselves.
            self._logger.info(line)
        except Exception as e:  # noqa: BLE001
            log.warning("audit log write failed (%r); event=%s lost", e, event)
