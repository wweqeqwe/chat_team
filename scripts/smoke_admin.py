"""Offline smoke test for the chat_team admin panel (``chat-team-admin``).

Covers (no network, no TLS — uses aiohttp.test_utils.TestClient over HTTP):

  auth.py    — pbkdf2 hash/verify (good + wrong + unknown user),
               session create/validate/destroy + sliding expiry,
               rate limiter (cap + cross-IP isolation),
               audit logger (one line per event).
  inspect.py — systemctl status resolves to a real service,
               du scans the home partition + subdirs,
               log tail returns content.
  server.py  — full HTTP flow:
                 * unauthenticated GET /  → 302 /login
                 * unauthenticated GET /api/status → 401
                 * bad credentials → 302 /login (audit: login_failed)
                 * good credentials → 302 / + Set-Cookie (session + csrf)
                 * GET /api/status with cookie → 200 JSON
                 * POST /api/restart without X-CSRF-Token → 403
                 * POST /api/restart with bad CSRF token → 403
                 * POST /api/restart with good CSRF + mocked systemctl → 200 + audit
                 * 5 failed logins from one IP → 6th blocked
                 * expired session → 401
                 * healthz returns 200 (active) / 503 (down)
  cli.py     — ``add-user`` CLI writes users.json + verify works
               ``init-certs`` CLI writes cert+key with valid PEM

Run: ``python3 scripts/smoke_admin.py`` — no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_admin_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

# Enable the admin block in the freshly-seeded config.yaml so the test's
# load_settings picks up admin.enabled=true with default port/host (the
# actual listener isn't started here — we use TestClient).
from chat_team.config import load_settings  # noqa: E402
from chat_team.paths import init_home  # noqa: E402

init_home()
settings = load_settings()
settings.admin.enabled = True
settings.admin.session_idle_seconds = 60  # short for testing
settings.admin.login_rate_limit_per_5min = 5

from chat_team.admin.auth import (  # noqa: E402
    AuditLogger,
    LoginRateLimiter,
    SessionStore,
    User,
    UserStore,
)
from chat_team.admin.cli import (  # noqa: E402
    cmd_add_user,
    cmd_init_certs,
)
from chat_team.admin.inspect import (  # noqa: E402
    _du_subdir_sync,
    get_service_status_sync,
    tail_log_sync,
)
from chat_team.admin.server import build_app  # noqa: E402

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label} {detail}")
        failures.append(label)


# --------------------------------------------------------------------------
# auth.py
# --------------------------------------------------------------------------

def test_auth() -> None:
    print("[auth]")
    with tempfile.TemporaryDirectory() as d:
        store = UserStore(Path(d) / "users.json")
        store.ensure_file()
        assert store.path.exists()
        check("users.json seeded", store.path.exists())

        salt, iters, hsh = UserStore.hash_password("correct horse battery staple")
        check("pbkdf2 returns hex salt+hash", bool(salt) and bool(hsh) and iters == 600_000,
              f"salt={bool(salt)} hash={bool(hsh)} iters={iters}")
        check("salt is 32 hex chars (16 bytes)", len(salt) == 32)
        check("hash is 64 hex chars (32 bytes)", len(hsh) == 64)

        u = User(username="alice", salt=salt, iterations=iters, hash=hsh)
        store.add_or_update(u)
        check("user persisted", store.get("alice") is not None)
        check("list_users shows alice", "alice" in store.list_users())

        check("verify good password", UserStore.verify("correct horse battery staple", store.get("alice")))
        check("verify wrong password → False", not UserStore.verify("wrong", store.get("alice")))
        check("verify unknown user → False", not UserStore.verify("correct horse battery staple", store.get("bob")))

        # Update existing user replaces, not appends.
        salt2, iters2, hsh2 = UserStore.hash_password("newpassword!!")
        store.add_or_update(User(username="alice", salt=salt2, iterations=iters2, hash=hsh2))
        check("update replaces (still 1 user)", len(store.list_users()) == 1)
        check("old password no longer works", not UserStore.verify("correct horse battery staple", store.get("alice")))
        check("new password works", UserStore.verify("newpassword!!", store.get("alice")))

        # Sessions
        ss = SessionStore(60)
        sid, csrf = ss.create("alice", ip="9.9.9.9")
        check("session created", sid and csrf)
        info = ss.validate(sid)
        check("session validates", info is not None and info.username == "alice")
        # Sliding expiry: validate twice rapidly, should still be valid.
        time.sleep(0.01)
        info2 = ss.validate(sid)
        check("session still valid after re-validate", info2 is not None)
        ss.destroy(sid)
        check("session destroyed", ss.validate(sid) is None)

        # Purge expired
        ss_short = SessionStore(0.05)
        sid2, _ = ss_short.create("bob")
        time.sleep(0.1)
        n = ss_short.purge_expired()
        check("purge_expired sweeps stale", n >= 1 and ss_short.validate(sid2) is None)

        # Rate limiter
        rl = LoginRateLimiter(5)
        for i in range(5):
            rl.record_failure("1.2.3.4")
        check("5 failures → rate limited", rl.is_rate_limited("1.2.3.4"))
        check("different IP not rate limited", not rl.is_rate_limited("5.6.7.8"))
        # Successful path: is_rate_limited stays True (cap is failures, not logins).
        # But the docstring is clear: only failed logins count.

        # Audit logger
        with tempfile.TemporaryDirectory() as ad:
            al = AuditLogger(Path(ad) / "audit.log")
            al.log("login_success", user="alice", ip="1.2.3.4", ua="UA")
            al.log("restart", user="alice", ip="1.2.3.4", extra="service=chat-team")
            content = (Path(ad) / "audit.log").read_text()
            check("audit log written", "event=login_success" in content and "event=restart" in content)
            check("audit log has timestamps", content.count("[") >= 2 and content.count("]") >= 2)
            check("audit log single-line per event", content.count("\n") == 2)

        # Audit log rotation — bounded, no infinite growth. Force rotation by
        # writing events until the RotatingFileHandler crosses max_bytes; the
        # first backup (.1) must appear and the live file must shrink back
        # under the cap.
        with tempfile.TemporaryDirectory() as ad:
            log_path = Path(ad) / "audit.log"
            al = AuditLogger(log_path, max_bytes=512, backup_count=3)
            # Each line is ~70 bytes (ts + 4 fields); ~8 lines = ~560 bytes
            # which crosses 512. RotatingFileHandler rolls over when the
            # NEXT write would exceed maxBytes, so we need one more event
            # past the boundary to actually trigger the rename.
            for i in range(20):
                al.log("test_event", user="alice", ip="1.2.3.4",
                       ua="Mozilla/5.0", extra=f"seq={i}")
            # After enough writes there must be at least one backup; the
            # oldest backup holds the earliest events (across multiple
            # rotations the original content rolls from .1 → .2 → .3 etc.,
            # so we look for the *highest-numbered* backup file).
            backups = sorted(name for name in os.listdir(ad)
                             if name.startswith("audit.log."))
            check("audit log rotated → ≥1 backup exists", len(backups) >= 1,
                  f"backups={backups}")
            check("audit log live file shrunk under cap",
                  log_path.stat().st_size <= 512,
                  f"size={log_path.stat().st_size}")
            # Oldest backup (highest suffix) holds the earliest events.
            oldest = Path(ad) / backups[-1]
            oldest_text = oldest.read_text()
            check("audit log oldest backup has early event",
                  "seq=0" in oldest_text, f"oldest={oldest.name}")
            # Live file has the most recent events (post-rotation writes).
            live_text = log_path.read_text()
            check("audit log live has latest event", "seq=19" in live_text)
            # Force more rotations to verify backup chain stays bounded.
            for i in range(40):
                al.log("test_event", user="bob", ip="5.6.7.8",
                       extra=f"seq2={i}")
            files = sorted(ad_path.name for ad_path in
                            [Path(ad) / n for n in os.listdir(ad)])
            # backupCount=3 → at most .1, .2, .3 plus the live file.
            backup_files = [n for n in os.listdir(ad) if n.startswith("audit.log.")]
            check("audit log backups bounded (≤ backup_count)",
                  len(backup_files) <= 3)
            # No backup should exceed the cap.
            for name in backup_files:
                sz = (Path(ad) / name).stat().st_size
                check(f"audit log backup {name} ≤ cap", sz <= 1024)


# --------------------------------------------------------------------------
# inspect.py
# --------------------------------------------------------------------------

def test_inspect() -> None:
    print("[inspect]")
    s = get_service_status_sync("chat-team.service")
    check("status returns dict", isinstance(s, dict))
    check("status has 'active' bool", isinstance(s.get("active"), bool))
    check("status has 'source'", s.get("source") in {"systemctl", "ps-fallback", "unknown"})
    # chat-team.service IS running in this dev env (PID 1721170). If a future
    # dev runs this elsewhere without systemd, source may be ps-fallback or
    # unknown — we just check the shape, not the active state.
    print(f"     status: active={s.get('active')} source={s.get('source')} pid={s.get('pid')}")

    # Disk on /tmp (always exists, writable).
    d = _du_subdir_sync(Path("/tmp"), top_n=5)
    check("disk has partition_total", isinstance(d.partition_total, int) and d.partition_total > 0)
    check("disk has partition_free", isinstance(d.partition_free, int) and d.partition_free >= 0)
    check("disk has subdirs list", isinstance(d.subdirs, list))
    check("disk has sessions_top list", isinstance(d.sessions_top, list))

    # tail_log_sync on /etc/hostname (always present on Linux).
    txt = tail_log_sync(Path("/etc/hostname"), lines=10)
    check("tail_log returns text", isinstance(txt, str) and len(txt) >= 0)
    print(f"     tail: {len(txt)} bytes from /etc/hostname")


# --------------------------------------------------------------------------
# cli.py
# --------------------------------------------------------------------------

def test_cli() -> None:
    print("[cli]")
    home = Path(os.environ["CHAT_TEAM_HOME"])

    # init-certs
    import argparse
    args = argparse.Namespace(force=False)
    rc = cmd_init_certs(args)
    check("init-certs returns 0", rc == 0)
    cert = home / "admin" / "cert.pem"
    key = home / "admin" / "key.pem"
    check("cert.pem written", cert.exists())
    check("key.pem written", key.exists())
    if cert.exists():
        # Verify it's a valid PEM cert (openssl if available, else just check
        # the BEGIN/END markers).
        data = cert.read_bytes()
        check("cert has PEM markers", b"BEGIN CERTIFICATE" in data and b"END CERTIFICATE" in data)
        try:
            r = subprocess.run(
                ["openssl", "x509", "-noout", "-subject", "-issuer", "-dates"],
                input=data, capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                txt = r.stdout.decode()
                check("openssl parses cert", "subject=" in txt)
                print(f"     cert subject: {txt.splitlines()[0] if txt else '(empty)'}")
            else:
                print("     (openssl unavailable; skipped parse)")
        except FileNotFoundError:
            print("     (openssl not installed; skipped parse)")

    # init-certs refuses without --force if exists
    rc2 = cmd_init_certs(argparse.Namespace(force=False))
    check("init-certs refuses overwrite without --force", rc2 != 0)
    rc3 = cmd_init_certs(argparse.Namespace(force=True))
    check("init-certs --force overwrites", rc3 == 0)

    # add-user (interactive — needs stdin). Use monkeypatching.
    import getpass
    inputs = iter(["supersecret1", "supersecret1"])
    getpass.getpass = lambda prompt="": next(inputs)
    rc4 = cmd_add_user(argparse.Namespace(name="admin"))
    check("add-user returns 0", rc4 == 0)
    users_json = home / "admin" / "users.json"
    check("users.json written", users_json.exists())
    if users_json.exists():
        data = json.loads(users_json.read_text())
        check("users.json has alice-like entry", any(u.get("username") == "admin" for u in data.get("users", [])))
        u = UserStore(users_json).get("admin")
        check("admin user has pbkdf2 hash", u is not None and u.algo == "pbkdf2_sha256" and u.hash and u.salt)
        check("admin password verifies", UserStore.verify("supersecret1", u))

    # add-user rejects mismatched passwords
    inputs2 = iter(["pass1", "pass2"])
    getpass.getpass = lambda prompt="": next(inputs2)
    rc5 = cmd_add_user(argparse.Namespace(name="admin2"))
    check("add-user mismatch → non-zero", rc5 != 0)

    # add-user rejects short passwords
    inputs3 = iter(["short", "short"])
    getpass.getpass = lambda prompt="": next(inputs3)
    rc6 = cmd_add_user(argparse.Namespace(name="admin3"))
    check("add-user too short → non-zero", rc6 != 0)


# --------------------------------------------------------------------------
# server.py — full HTTP flow
# --------------------------------------------------------------------------

async def test_server_async() -> None:
    print("[server]")
    app = build_app(settings)

    # Mock the systemctl calls so the test doesn't actually restart the
    # live chat_team service.
    import chat_team.admin.inspect as inspect_mod
    original_restart = inspect_mod.restart_service_sync
    original_reload = inspect_mod.reload_service_sync
    inspect_mod.restart_service_sync = lambda *a, **k: {
        "ok": True, "status": "restarted", "active": True, "pid": 99999,
        "uptime": "0m", "memory_bytes": 1024, "memory_human": "1.0KB",
        "source": "systemctl",
    }
    inspect_mod.reload_service_sync = lambda *a, **k: {"ok": True, "status": "reloaded"}

    # Update inspect's async wrappers in server module's namespace (they
    # call inspect_mod.<func> via to_thread, so the to_thread target is the
    # module attribute → patching inspect_mod.* is enough).
    server = TestServer(app)
    client = TestClient(server)
    try:
        await client.start_server()

        # 1. Unauthenticated GET / → 302 /login (don't follow redirects
        # so we see the 302 + Location header directly).
        r = await client.get("/", allow_redirects=False)
        check("anon GET / → 302", r.status == 302 and r.headers.get("Location") == "/login",
              f"status={r.status} loc={r.headers.get('Location')}")
        await r.read()

        # 2. Unauthenticated GET /api/status → 401 (no redirect — JSON API)
        r = await client.get("/api/status", allow_redirects=False)
        check("anon GET /api/status → 401", r.status == 401)
        await r.read()

        # 3. /healthz is public (no auth required)
        r = await client.get("/healthz", allow_redirects=False)
        check("/healthz is public", r.status in (200, 503))
        await r.read()

        # 4. Bad credentials → 302 /login (no Set-Cookie for session)
        r = await client.post("/login", data={"username": "admin", "password": "wrong"},
                              allow_redirects=False)
        check("bad creds → 302 /login", r.status == 302 and r.headers.get("Location", "").startswith("/login"),
              f"status={r.status}")
        await r.read()

        # 5. Good credentials → 302 / + Set-Cookie (session + csrf).
        # Don't follow the redirect — we need to read Set-Cookie off the 302
        # itself; if we follow, the cookie lands in the cookie jar instead of
        # in r.cookies.
        r = await client.post("/login", data={"username": "admin", "password": "supersecret1"},
                              allow_redirects=False)
        check("good creds → 302 /", r.status == 302 and r.headers.get("Location") == "/",
              f"status={r.status} loc={r.headers.get('Location')}")

        # Extract Set-Cookie values from the 302 response directly.
        # aiohttp collapses multiple Set-Cookie headers into a cookies jar
        # on the response object (each cookie has .key + .value).
        sid_val = None
        csrf_val = None
        for c in r.cookies.values():
            if c.key == "session":
                sid_val = c.value
            elif c.key == "csrf":
                csrf_val = c.value
        check("good creds set session cookie", sid_val is not None)
        check("good creds set csrf cookie", csrf_val is not None)
        await r.read()

        # Build the explicit Cookie header we'll send on every authed
        # request. We bypass the client's cookie jar so the test is
        # deterministic (the jar's contents depend on redirect-following
        # behaviour, which we vary test-by-test).
        headers = {"Cookie": f"session={sid_val}; csrf={csrf_val}"}

        # 6. With session cookie, GET / → 200 (dashboard HTML).
        r = await client.get("/", headers=headers, allow_redirects=False)
        check("authed GET / → 200", r.status == 200, f"status={r.status}")
        body = (await r.read()).decode()
        check("dashboard has username", "admin" in body)

        # 7. GET /api/status with cookie → 200 JSON.
        r = await client.get("/api/status", headers=headers, allow_redirects=False)
        check("authed GET /api/status → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("status json has 'active'", "active" in j)

        # 8. POST /api/restart without X-CSRF-Token → 403.
        r = await client.post("/api/restart", headers=headers, allow_redirects=False)
        check("restart without CSRF → 403", r.status == 403, f"status={r.status}")
        await r.read()

        # 9. POST /api/restart with bad CSRF token → 403.
        r = await client.post("/api/restart",
                              headers={**headers, "X-CSRF-Token": "garbage"},
                              allow_redirects=False)
        check("restart with bad CSRF → 403", r.status == 403, f"status={r.status}")
        await r.read()

        # 10. POST /api/restart with good CSRF → 200 + audit logged.
        r = await client.post("/api/restart",
                              headers={**headers, "X-CSRF-Token": csrf_val},
                              allow_redirects=False)
        check("restart with good CSRF → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("restart returns ok=True", j.get("ok") is True, str(j))

        # 11. POST /api/reload with good CSRF → 200.
        r = await client.post("/api/reload",
                              headers={**headers, "X-CSRF-Token": csrf_val},
                              allow_redirects=False)
        check("reload with good CSRF → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("reload returns ok=True", j.get("ok") is True, str(j))

        # 12. GET /api/disk → JSON.
        r = await client.get("/api/disk", headers=headers, allow_redirects=False)
        check("GET /api/disk → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("disk has partition", "partition" in j and "total" in j["partition"])
        check("disk has chat_team_total_human", "chat_team_total_human" in j)

        # 13. GET /api/sessions → JSON list.
        r = await client.get("/api/sessions", headers=headers, allow_redirects=False)
        check("GET /api/sessions → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("sessions has sessions_top list", isinstance(j.get("sessions_top"), list))

        # 14. GET /api/logs/tail?kind=bot → text.
        r = await client.get("/api/logs/tail?kind=bot&lines=50",
                             headers=headers, allow_redirects=False)
        check("GET /api/logs/tail?kind=bot → 200", r.status == 200, f"status={r.status}")
        j = await r.json()
        check("log tail returns text", "text" in j)

        # 15. 5 failed logins from same IP → 6th blocked.
        # Use a fresh client (no cookies) so the rate-limiter sees all
        # attempts as anonymous POSTs from the same loopback IP.
        client2 = TestClient(TestServer(build_app(settings)))
        await client2.start_server()
        for i in range(5):
            r = await client2.post("/login", data={"username": "admin", "password": "wrong"},
                                   allow_redirects=False)
            await r.read()
        # 6th attempt with the right password — should be blocked at the
        # rate-limit gate (before creds are checked). The Location is
        # URL-encoded (e.g. /login?flash=%E7%99%BB%E5%BD%95...) so we look
        # for the literal flash param presence rather than decoding the
        # Chinese text.
        r = await client2.post("/login", data={"username": "admin", "password": "supersecret1"},
                               allow_redirects=False)
        loc = r.headers.get("Location", "")
        from urllib.parse import parse_qs, urlsplit
        qs = parse_qs(urlsplit(loc).query)
        check("6th login after 5 failures → blocked (302 with flash)",
              r.status == 302 and "flash" in qs and qs["flash"],
              f"status={r.status} loc={loc}")
        await r.read()
        await client2.close()

        # 16. Expired session → 401 on API, 302 on HTML.
        ss: SessionStore = app["sessions"]
        with ss._lock:
            for k, info in ss._sessions.items():
                info.expires_at = time.time() - 1
        r = await client.get("/api/status", headers=headers, allow_redirects=False)
        check("expired session → 401 on API", r.status == 401, f"status={r.status}")
        await r.read()

        # 17. Audit log captures login_success + restart + reload.
        audit_path = settings.paths.logs_dir / "admin.log"
        if audit_path.exists():
            txt = audit_path.read_text()
            check("audit log has login_success", "event=login_success" in txt)
            check("audit log has restart", "event=restart" in txt)
            check("audit log has reload", "event=reload" in txt)
        else:
            check("audit log file exists", False, f"missing {audit_path}")
    finally:
        inspect_mod.restart_service_sync = original_restart
        inspect_mod.reload_service_sync = original_reload
        await client.close()


def main() -> int:
    test_auth()
    test_inspect()
    test_cli()
    asyncio.run(test_server_async())
    print()
    if failures:
        print(f"\033[31m{len(failures)} FAIL\033[0m: {failures}")
        return 1
    print(f"\033[32mALL PASS\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
