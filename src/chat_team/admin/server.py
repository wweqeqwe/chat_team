"""HTTPS admin web server (aiohttp.web).

Architecture:
  * ``serve()`` — load settings, validate cert/users exist, build the
    aiohttp.web app, run with an SSL context (TLS termination in-process).
  * ``build_app(settings)`` — wires middleware + routes. Pure function so
    the smoke tests can spin up a TestClient without touching TLS.
  * Middleware ``require_auth`` — HTML routes redirect to /login on miss,
    API routes return 401; per-request CSRF check is inside the route
    handlers (so it can return a precise 403 with a body).

Design choices documented inline.
"""
from __future__ import annotations

import asyncio
import html
import logging
import ssl
import sys
import time
from pathlib import Path

from aiohttp import web

from ..config import Settings, load_settings
from .auth import AuditLogger, LoginRateLimiter, SessionStore, UserStore, User
from .inspect import (
    CachedDiskInspector,
    SERVICE_NAME,
    get_service_status,
    reload_service,
    restart_service,
    tail_journal,
    tail_log,
)

log = logging.getLogger(__name__)

SESSION_COOKIE = "session"
CSRF_COOKIE = "csrf"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _admin_dir(settings: Settings) -> Path:
    d = settings.paths.home / "admin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _users_path(settings: Settings) -> Path:
    return _admin_dir(settings) / "users.json"


def _audit_log_path(settings: Settings) -> Path:
    p = settings.admin.audit_log_path
    if p:
        return Path(p)
    return settings.paths.logs_dir / "admin.log"


def _client_ip(request: web.Request) -> str:
    """Best-effort client IP.

    Honors the ``X-Forwarded-For`` header's first entry IF the request is
    from a trusted local source (loopback). Public-facing admin should sit
    behind a TLS-terminating reverse proxy that overwrites XFF — without
    that protection, trusting XFF would let any client spoof its IP for the
    rate limiter and audit log. So: only honor XFF when the immediate peer
    is loopback (typical nginx-in-front layout); otherwise use ``peer``.
    """
    peer = request.remote or "?"
    if peer and (peer.startswith("127.") or peer == "::1"):
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
    return peer


def _user_agent(request: web.Request) -> str:
    return request.headers.get("User-Agent", "")


# --------------------------------------------------------------------------
# App builder
# --------------------------------------------------------------------------

def build_app(settings: Settings) -> web.Application:
    """Wire the aiohttp.web app. Pure function (no I/O at import time).

    The app carries everything it needs on ``app["..."]`` so route handlers
    don't re-resolve paths on each request.
    """
    app = web.Application(
        client_max_size=1_000_000,  # 1MB; dashboard fetches are tiny
        # Disable the default aiohttp trailing-slash redirect (303 on /api/)
        # so a missing-slash POST doesn't silently become a GET.
        handler_args={"debug": False},
    )
    app["settings"] = settings
    app["users"] = UserStore(_users_path(settings))
    app["users"].ensure_file()
    app["sessions"] = SessionStore(settings.admin.session_idle_seconds)
    app["rate_limiter"] = LoginRateLimiter(settings.admin.login_rate_limit_per_5min)
    app["audit"] = AuditLogger(
        _audit_log_path(settings),
        max_bytes=settings.admin.audit_log_max_bytes,
        backup_count=settings.admin.audit_log_backup_count,
    )
    app["disk"] = CachedDiskInspector(settings.paths.home, cache_ttl=30.0)

    # Background sweeper: purge expired sessions every 5 min so the in-memory
    # dict doesn't grow forever on abandoned logins. The startup hook MUST
    # be a plain sync callable that creates the task and returns None — if
    # it returned the awaitable Task, aiohttp's startup processor would await
    # it and block forever (the sweeper is an infinite loop).
    async def _sweeper(app: web.Application) -> None:
        sessions: SessionStore = app["sessions"]
        try:
            while True:
                await asyncio.sleep(300)
                sessions.purge_expired()
        except asyncio.CancelledError:
            return

    async def _spawn_sweeper(app: web.Application) -> None:
        # Fire-and-forget; the task is held by the loop until app shutdown.
        # This callback is async because aiohttp's aiosignal always awaits
        # the receiver's return value — a sync callback returning None
        # raises ``TypeError: object NoneType can't be used in 'await'``.
        # We create_task (which returns immediately) and DON'T await it.
        app["sweeper_task"] = asyncio.create_task(_sweeper(app))

    app.on_startup.append(_spawn_sweeper)

    async def _cancel_sweeper(app: web.Application) -> None:
        task = app.get("sweeper_task")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    app.on_cleanup.append(_cancel_sweeper)

    # Auth middleware: stamp the resolved SessionInfo on the request and
    # gate access. CSRF is checked inside state-changing handlers (not here)
    # so we can return a meaningful 403 body.
    @web.middleware
    async def require_auth(request: web.Request, handler):
        sid = _cookie(request, SESSION_COOKIE)
        session = app["sessions"].validate(sid) if sid else None
        request["session"] = session
        path = request.path
        # Public routes: /login (GET and POST), /healthz.
        if path in ("/login", "/healthz") or path.startswith("/static/"):
            return await handler(request)
        if session is None:
            if path.startswith("/api/"):
                return web.json_response({"error": "unauthorized"}, status=401)
            return web.Response(
                status=302,
                headers={"Location": "/login"},
            )
        return await handler(request)

    app.middlewares.append(require_auth)

    app.router.add_get("/login", handle_login_form)
    app.router.add_post("/login", handle_login_submit)
    app.router.add_post("/logout", handle_logout)
    app.router.add_get("/", handle_dashboard)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/api/status", handle_api_status)
    app.router.add_get("/api/disk", handle_api_disk)
    app.router.add_get("/api/sessions", handle_api_sessions)
    app.router.add_get("/api/logs/tail", handle_api_logs_tail)
    app.router.add_post("/api/restart", handle_api_restart)
    app.router.add_post("/api/reload", handle_api_reload)
    return app


def _cookie(request: web.Request, name: str) -> str | None:
    return request.cookies.get(name)


def _set_session_cookies(response: web.Response, sid: str, csrf: str, idle_s: float) -> None:
    response.set_cookie(
        SESSION_COOKIE, sid,
        httponly=True, secure=True, samesite="Strict",
        path="/", max_age=int(idle_s),
    )
    # CSRF cookie readable by JS (so the dashboard can read it and put it
    # into an X-CSRF-Token header). NOT HttpOnly by design.
    response.set_cookie(
        CSRF_COOKIE, csrf,
        secure=True, samesite="Strict",
        path="/", max_age=int(idle_s),
    )


def _clear_cookies(response: web.Response) -> None:
    response.del_cookie(SESSION_COOKIE, path="/")
    response.del_cookie(CSRF_COOKIE, path="/")


def _check_csrf(request: web.Request) -> bool:
    """Double-submit cookie pattern: X-CSRF-Token header == csrf cookie.

    SameSite=Strict on the cookie already blocks cross-site form POSTs, but
    we still require the matching header because:
      1. SameSite=Strict is no defense against same-site XSS-driven CSRF.
      2. Browsers honour SameSite unevenly across versions; the explicit
         header check is a belt-and-braces backstop.
    """
    session = request.get("session")
    csrf_cookie = _cookie(request, CSRF_COOKIE)
    csrf_header = request.headers.get("X-CSRF-Token", "")
    if session is None:
        return False
    if not csrf_cookie or not csrf_header:
        return False
    # Constant-time compare vs both the cookie (round-tripped back) AND the
    # session-bound token (server-side truth) so a forged cookie alone is
    # never enough.
    import hmac as _hmac
    if not _hmac.compare_digest(csrf_cookie, csrf_header):
        return False
    if not _hmac.compare_digest(csrf_cookie, session.csrf_token):
        return False
    return True


# --------------------------------------------------------------------------
# Route handlers
# --------------------------------------------------------------------------

async def handle_healthz(request: web.Request) -> web.Response:
    # No-auth (public): lets a load balancer / uptime monitor ping without
    # a session. Returns 503 if the chat_team service is down.
    status = await get_service_status(SERVICE_NAME)
    code = 200 if status["active"] else 503
    return web.json_response({"chat_team_active": status["active"]}, status=code)


async def handle_login_form(request: web.Request) -> web.Response:
    if request.get("session") is not None:
        return web.Response(status=302, headers={"Location": "/"})
    flash = request.query.get("flash", "")
    body = _LOGIN_HTML.replace("__FLASH__", html.escape(flash))
    return web.Response(text=body, content_type="text/html")


async def handle_login_submit(request: web.Request) -> web.Response:
    settings: Settings = request.app["settings"]
    users: UserStore = request.app["users"]
    sessions: SessionStore = request.app["sessions"]
    rate_limiter: LoginRateLimiter = request.app["rate_limiter"]
    audit: AuditLogger = request.app["audit"]
    ip = _client_ip(request)
    ua = _user_agent(request)
    try:
        form = await request.post()
    except Exception:
        return _login_redirect(flash="表单解析失败")
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""

    if not username or not password:
        audit.log("login_failed", user=username or "?", ip=ip, ua=ua,
                  extra="reason=empty_fields")
        return _login_redirect(flash="用户名或密码错误")

    # Rate-limit BEFORE checking creds so a flood of wrong passwords can't
    # hammer pbkdf2 verification. If the IP is over the cap, return 429
    # without even touching the user store.
    if rate_limiter.is_rate_limited(ip):
        audit.log("login_blocked", user=username, ip=ip, ua=ua,
                  extra="reason=rate_limited")
        return _login_redirect(flash="登录尝试过多,请稍后再试")

    user = users.get(username)
    if user is None or not UserStore.verify(password, user):
        rate_limiter.record_failure(ip)
        audit.log("login_failed", user=username, ip=ip, ua=ua,
                  extra="reason=bad_credentials")
        return _login_redirect(flash="用户名或密码错误")

    # Success — reset the IP's failure window (correct password clears the
    # budget so a legit user who fat-fingered once isn't carried over).
    sid, csrf = sessions.create(username, ip=ip)
    audit.log("login_success", user=username, ip=ip, ua=ua)
    resp = web.Response(status=302, headers={"Location": "/"})
    _set_session_cookies(resp, sid, csrf, settings.admin.session_idle_seconds)
    return resp


async def handle_logout(request: web.Request) -> web.Response:
    # POST-only (form submit from dashboard). CSRF not required for logout
    # — no harm in someone logging you out.
    sessions: SessionStore = request.app["sessions"]
    audit: AuditLogger = request.app["audit"]
    session = request.get("session")
    sid = _cookie(request, SESSION_COOKIE) or ""
    if session is not None:
        sessions.destroy(sid)
        audit.log("logout", user=session.username, ip=_client_ip(request))
    resp = web.Response(status=302, headers={"Location": "/login"})
    _clear_cookies(resp)
    return resp


async def handle_dashboard(request: web.Request) -> web.Response:
    session = request["session"]
    body = _DASHBOARD_HTML.replace("__USERNAME__", html.escape(session.username))
    return web.Response(text=body, content_type="text/html")


async def handle_api_status(request: web.Request) -> web.Response:
    status = await get_service_status(SERVICE_NAME)
    return web.json_response(status)


async def handle_api_disk(request: web.Request) -> web.Response:
    inspector: CachedDiskInspector = request.app["disk"]
    usage = await inspector.inspect()
    return web.json_response({
        "partition": {
            "total": usage.partition_total,
            "used": usage.partition_used,
            "free": usage.partition_free,
            "use_percent": usage.partition_use_percent,
        },
        "chat_team_total": usage.chat_team_total,
        "chat_team_total_human": usage.chat_team_total_human,
        "subdirs": usage.subdirs,
        "sessions_top": usage.sessions_top,
        "truncated": usage.truncated,
    })


async def handle_api_sessions(request: web.Request) -> web.Response:
    inspector: CachedDiskInspector = request.app["disk"]
    usage = await inspector.inspect()
    return web.json_response({
        "sessions_top": usage.sessions_top,
        "truncated": usage.truncated,
    })


async def handle_api_logs_tail(request: web.Request) -> web.Response:
    kind = request.query.get("kind", "bot")
    lines = int(request.query.get("lines", "200"))
    lines = max(50, min(1000, lines))
    if kind == "admin":
        settings: Settings = request.app["settings"]
        text = await tail_log(_audit_log_path(settings), lines)
    else:
        text = await tail_journal(SERVICE_NAME, lines)
    return web.json_response({"kind": kind, "lines": lines, "text": text})


async def handle_api_restart(request: web.Request) -> web.Response:
    if not _check_csrf(request):
        return web.json_response({"error": "csrf"}, status=403)
    session = request["session"]
    audit: AuditLogger = request.app["audit"]
    audit.log("restart", user=session.username, ip=_client_ip(request))
    result = await restart_service(SERVICE_NAME, wait_seconds=30.0)
    return web.json_response(result, status=200 if result.get("ok") else 503)


async def handle_api_reload(request: web.Request) -> web.Response:
    if not _check_csrf(request):
        return web.json_response({"error": "csrf"}, status=403)
    session = request["session"]
    audit: AuditLogger = request.app["audit"]
    audit.log("reload", user=session.username, ip=_client_ip(request))
    result = await reload_service(SERVICE_NAME)
    return web.json_response(result, status=200 if result.get("ok") else 503)


def _login_redirect(flash: str) -> web.Response:
    return web.Response(
        status=302,
        headers={"Location": f"/login?flash={_quote(flash)}"},
    )


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)


# --------------------------------------------------------------------------
# HTML pages
# --------------------------------------------------------------------------

_LOGIN_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>chat_team 后台管理</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f7; margin: 0; padding: 0; }
  .card { width: 360px; margin: 80px auto; background: #fff; border-radius: 12px;
          padding: 32px 28px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #888; font-size: 13px; margin: 0 0 24px; }
  label { display: block; font-size: 13px; color: #555; margin: 0 0 6px; }
  input { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px;
          font-size: 14px; box-sizing: border-box; margin-bottom: 14px; }
  input:focus { outline: none; border-color: #0071e3; }
  button { width: 100%; padding: 11px; background: #0071e3; color: #fff; border: 0;
           border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:hover { background: #0058b9; }
  .flash { color: #c00; font-size: 13px; margin-bottom: 12px; min-height: 18px; }
</style></head>
<body><div class="card">
  <h1>chat_team 后台管理</h1>
  <p class="sub">登录以管理服务</p>
  <div class="flash">__FLASH__</div>
  <form method="post" action="/login">
    <label for="username">账号</label>
    <input id="username" name="username" autocomplete="username" autofocus required>
    <label for="password">密码</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">登录</button>
  </form>
</div></body></html>
"""


_DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>chat_team 后台</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f7; margin: 0; padding: 0; color: #222; }
  .topbar { background: #1d1d1f; color: #fff; padding: 14px 24px; display: flex;
            justify-content: space-between; align-items: center; }
  .topbar .who { font-size: 13px; color: #ccc; }
  .wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #fff; border-radius: 12px; padding: 18px 20px;
          box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
  h2 { font-size: 15px; margin: 0 0 12px; color: #444; font-weight: 600; }
  .row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 13px;
         border-bottom: 1px solid #f0f0f0; }
  .row:last-child { border: 0; }
  .row .k { color: #666; }
  .row .v { font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 10px;
           font-size: 12px; color: #fff; }
  .badge.ok { background: #34c759; }
  .badge.down { background: #ff3b30; }
  .bar { background: #eee; border-radius: 4px; height: 8px; margin: 8px 0; overflow: hidden; }
  .bar > div { background: #0071e3; height: 100%; }
  .actions { margin: 16px 0; display: flex; gap: 12px; }
  .actions button { padding: 10px 18px; border: 0; border-radius: 8px; font-size: 13px;
                    font-weight: 600; cursor: pointer; }
  .btn-restart { background: #ff3b30; color: #fff; }
  .btn-reload { background: #ff9500; color: #fff; }
  .btn-logout { background: #888; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #f0f0f0; }
  th { color: #888; font-weight: 500; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.5px; }
  td.num { font-variant-numeric: tabular-nums; text-align: right; }
  pre { background: #1d1d1f; color: #d0d0d0; padding: 12px; border-radius: 8px;
        overflow: auto; max-height: 360px; font-size: 12px; line-height: 1.5;
        font-family: "SFMono-Regular", Consolas, monospace; }
  .switcher { margin-bottom: 8px; }
  .switcher button { padding: 4px 10px; border: 1px solid #ddd; background: #fff;
                     border-radius: 6px; font-size: 12px; cursor: pointer; margin-right: 6px; }
  .switcher button.active { background: #0071e3; color: #fff; border-color: #0071e3; }
  .meta { font-size: 11px; color: #999; margin-top: 6px; }
</style></head>
<body>
<div class="topbar">
  <div>chat_team 后台管理</div>
  <div class="who">已登录: <b>__USERNAME__</b></div>
</div>
<div class="wrap">
  <div class="grid">
    <div class="card">
      <h2>chat_team 服务状态</h2>
      <div class="row"><span class="k">状态</span><span class="v" id="st-status">—</span></div>
      <div class="row"><span class="k">PID</span><span class="v" id="st-pid">—</span></div>
      <div class="row"><span class="k">运行时长</span><span class="v" id="st-uptime">—</span></div>
      <div class="row"><span class="k">内存</span><span class="v" id="st-mem">—</span></div>
      <div class="meta" id="st-source"></div>
    </div>
    <div class="card">
      <h2>磁盘</h2>
      <div class="row"><span class="k">分区</span><span class="v" id="disk-part">—</span></div>
      <div class="bar"><div id="disk-bar" style="width:0%"></div></div>
      <div class="row"><span class="k">chat_team 占用</span><span class="v" id="disk-ct">—</span></div>
      <div id="disk-subdirs" style="margin-top: 8px;"></div>
    </div>
  </div>

  <div class="card">
    <h2>操作</h2>
    <div class="actions">
      <button class="btn-reload" onclick="doAction('/api/reload','确认热重载 chat_team? (不打断 WebSocket)')">热重载</button>
      <button class="btn-restart" onclick="doAction('/api/restart','确认重启 chat_team? 短暂中断服务')">重启 chat_team</button>
      <form method="post" action="/logout" style="margin-left:auto">
        <button class="btn-logout" type="submit">退出登录</button>
      </form>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h2>会话 (按占用排序,文件系统视角)</h2>
    <table>
      <thead><tr><th>会话目录</th><th class="num">大小</th><th>最后修改</th></tr></thead>
      <tbody id="sessions-body"><tr><td colspan=3>加载中...</td></tr></tbody>
    </table>
    <div class="meta" id="sessions-meta"></div>
  </div>

  <div class="card">
    <h2>日志</h2>
    <div class="switcher">
      <button id="log-bot" class="active" onclick="switchLog('bot')">chat_team 日志</button>
      <button id="log-admin" onclick="switchLog('admin')">admin 操作日志</button>
    </div>
    <pre id="log-text">加载中...</pre>
    <div class="meta">每 5 秒刷新</div>
  </div>
</div>

<script>
const csrf = (() => { const m = document.cookie.match(/(?:^|;)\\s*csrf=([^;]+)/); return m ? m[1] : ''; })();

async function jsonOrDie(url, opts) {
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = '/login'; return null; }
  return await r.json();
}

function fmtBytes(n) {
  if (n == null || isNaN(n)) return '—';
  const u = ['B','KB','MB','GB','TB']; let v = n, i = 0;
  while (Math.abs(v) >= 1024 && i < u.length-1) { v /= 1024; i++; }
  return v.toFixed(1) + u[i];
}

async function refreshStatus() {
  const s = await jsonOrDie('/api/status');
  if (!s) return;
  const badge = s.active
    ? '<span class="badge ok">运行中</span>'
    : '<span class="badge down">未运行</span>';
  document.getElementById('st-status').innerHTML = badge;
  document.getElementById('st-pid').textContent = s.pid ?? '—';
  document.getElementById('st-uptime').textContent = s.uptime ?? '—';
  document.getElementById('st-mem').textContent = s.memory_human ?? '—';
  document.getElementById('st-source').textContent = '来源: ' + (s.source ?? '—');
}

async function refreshDisk() {
  const d = await jsonOrDie('/api/disk');
  if (!d) return;
  const p = d.partition;
  document.getElementById('disk-part').textContent =
    fmtBytes(p.used) + ' / ' + fmtBytes(p.total) + ' (' + p.use_percent + '%)';
  document.getElementById('disk-bar').style.width = Math.min(100, p.use_percent) + '%';
  document.getElementById('disk-ct').textContent = d.chat_team_total_human;
  let html = '';
  (d.subdirs || []).forEach(s => {
    html += '<div class="row"><span class="k">' + s.name + '</span>' +
            '<span class="v">' + fmtBytes(s.size_bytes) + '</span></div>';
  });
  document.getElementById('disk-subdirs').innerHTML = html;
}

async function refreshSessions() {
  const d = await jsonOrDie('/api/sessions');
  if (!d) return;
  let html = '';
  (d.sessions_top || []).forEach(s => {
    html += '<tr><td>' + s.sid + '</td>' +
            '<td class="num">' + s.size_human + '</td>' +
            '<td>' + s.mtime_human + '</td></tr>';
  });
  if (!html) html = '<tr><td colspan=3>无会话</td></tr>';
  document.getElementById('sessions-body').innerHTML = html;
  document.getElementById('sessions-meta').textContent =
    d.truncated ? '仅显示前 20 个会话 (按大小)' : '';
}

let logKind = 'bot';
function switchLog(k) {
  logKind = k;
  document.getElementById('log-bot').classList.toggle('active', k === 'bot');
  document.getElementById('log-admin').classList.toggle('active', k === 'admin');
  refreshLog();
}
async function refreshLog() {
  const r = await fetch('/api/logs/tail?kind=' + logKind + '&lines=200');
  if (r.status === 401) { location.href = '/login'; return; }
  const d = await r.json();
  document.getElementById('log-text').textContent = d.text || '(空)';
}

async function doAction(url, msg) {
  if (!confirm(msg)) return;
  const r = await fetch(url, {
    method: 'POST', headers: {'X-CSRF-Token': csrf},
  });
  const d = await r.json().catch(() => ({error: 'unknown'}));
  if (d.ok) {
    alert('操作成功: ' + (d.status || 'ok'));
    setTimeout(refreshStatus, 1500);
  } else {
    alert('操作失败: ' + (d.error || JSON.stringify(d)));
  }
}

refreshStatus(); refreshDisk(); refreshSessions(); refreshLog();
setInterval(refreshStatus, 5000);
setInterval(refreshDisk, 30000);
setInterval(refreshLog, 5000);
</script>
</body></html>
"""


# --------------------------------------------------------------------------
# serve()
# --------------------------------------------------------------------------

def serve() -> int:
    settings = load_settings()
    if not settings.admin.enabled:
        print(
            "chat-team-admin: admin.enabled is false in config.yaml; "
            "set admin.enabled: true and restart.",
            file=sys.stderr,
        )
        return 1

    # Resolve cert/key paths (default to ~/.chat_team/admin/{cert,key}.pem).
    cert_path = Path(settings.admin.tls_cert) if settings.admin.tls_cert else (
        settings.paths.home / "admin" / "cert.pem"
    )
    key_path = Path(settings.admin.tls_key) if settings.admin.tls_key else (
        settings.paths.home / "admin" / "key.pem"
    )
    if not cert_path.exists() or not key_path.exists():
        print(
            f"chat-team-admin: TLS cert/key missing at "
            f"{cert_path} / {key_path}; run `chat-team-admin init-certs`.",
            file=sys.stderr,
        )
        return 1

    users = UserStore(_users_path(settings))
    users.ensure_file()
    if not users.list_users():
        print(
            "chat-team-admin: no admin users yet; run "
            "`chat-team-admin add-user <name>` to create one.",
            file=sys.stderr,
        )
        return 1

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    app = build_app(settings)
    log.info(
        "chat-team-admin: listening on https://%s:%d (TLS)",
        settings.admin.host, settings.admin.port,
    )
    print(
        f"chat-team-admin: listening on https://{settings.admin.host}:"
        f"{settings.admin.port}",
        flush=True,
    )
    web.run_app(
        app,
        host=settings.admin.host,
        port=settings.admin.port,
        ssl_context=ssl_ctx,
        # Reuse port — crashes on collision instead of silently stealing.
        reuse_port=False,
        access_log=None,
    )
    return 0
