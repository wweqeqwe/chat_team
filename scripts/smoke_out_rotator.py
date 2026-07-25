"""Smoke test for ``chat_team.out_rotator.OutFileRotator``.

The daemon's stdout/stderr are dup2'd to ``~/.chat_team/logs/chat_team.out``
via ``os.open(path, O_WRONLY|O_CREAT|O_APPEND)`` in
``chat_team.daemon.daemonize_and_run``. Because the FD is held open at the
OS level, a Python ``RotatingFileHandler`` can't intercept it — without
``OutFileRotator`` the file grows unbounded.

The reaper uses a copytruncate strategy:
  1. shift .N → .N+1 (oldest deleted),
  2. shutil.copy2(live, .1),
  3. os.truncate(live, 0).

The open O_APPEND FD stays valid: every ``write(2)`` seeks to the current
end-of-file before writing, so after truncation writes resume at byte 0
of the same inode. This is the property we test most carefully here — a
real FD opened O_APPEND and held open during rotation must keep writing
into the (now-truncated) live file rather than into the .1 backup.

Run: ``python3 scripts/smoke_out_rotator.py`` — no network, no LLM.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_out_rotator_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.out_rotator import OutFileRotator

_passed = 0
_failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  \033[32mPASS\033[0m {name}")
    else:
        _failed += 1
        print(f"  \033[31mFAIL\033[0m {name}  {detail}")


def _write_oappend(path: Path, data: bytes) -> int:
    """Open a file O_APPEND the way daemon.py does and write ``data``.

    Returns the open fd (caller closes it). Mimics the dup2 target —
    rotation must preserve this fd's writes.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.write(fd, data)
    return fd


async def test_basic_rotation() -> None:
    print("[basic rotation]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "chat_team.out"
        # Pre-populate beyond cap so the first sweep rotates immediately.
        live.write_bytes(b"x" * 1500)
        rot = OutFileRotator(
            live, max_bytes=1024, backup_count=3,
            check_interval_seconds=10.0,
        )
        # _rotate_now is sync — call directly to avoid the asyncio loop's
        # sleep interval. This is the same method the background task calls
        # via to_thread.
        rot._rotate_now()

        check("live file truncated under cap",
              live.stat().st_size == 0,
              f"size={live.stat().st_size}")
        check("backup .1 created with old contents",
              live.with_suffix(".out.1").exists() and
              live.with_suffix(".out.1").stat().st_size == 1500)


async def test_backup_chain_bounded() -> None:
    print("[backup chain stays bounded]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "chat_team.out"
        rot = OutFileRotator(
            live, max_bytes=100, backup_count=3,
            check_interval_seconds=10.0,
        )
        # Drive several rotations past backup_count to ensure the chain
        # never exceeds backup_count files.
        for i in range(8):
            live.write_bytes(b"Y" * 200)  # well above the 100-byte cap
            rot._rotate_now()
        backups = sorted(name for name in os.listdir(td) if name.startswith("chat_team.out."))
        check("backups bounded to backup_count=3", len(backups) == 3,
              f"backups={backups}")
        # Names must be .1, .2, .3 — no gaps, no .4+.
        expected = {"chat_team.out.1", "chat_team.out.2", "chat_team.out.3"}
        check("backup names are .1/.2/.3", set(backups) == expected,
              f"got={set(backups)}")


async def test_oappend_fd_survives_truncation() -> None:
    """The single most important property: an open O_APPEND FD held during
    rotation must keep writing into the live file (now truncated), NOT into
    the .1 backup that was just shutil.copy2'd from it.
    """
    print("[O_APPEND FD survives truncation]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "chat_team.out"
        # 1. Open FD the daemon way, write some bytes (under cap so no rotation).
        fd = _write_oappend(live, b"BEFORE-ROTATION\n")
        try:
            rot = OutFileRotator(
                live, max_bytes=10, backup_count=2,
                check_interval_seconds=10.0,
            )
            # 2. Rotate (force-by-pre-writing more than cap above + here).
            # The live file currently has ~16 bytes, cap is 10 → rotate.
            rot._rotate_now()
            check("live truncated to 0 after rotation",
                  live.stat().st_size == 0,
                  f"size={live.stat().st_size}")
            check(".1 backup holds pre-rotation content",
                  b"BEFORE-ROTATION" in live.with_suffix(".out.1").read_bytes())

            # 3. Write more through the SAME fd (the daemon's stdout/stderr).
            os.write(fd, b"AFTER-ROTATION\n")

            # 4. The new content must be in the live file, not in .1.
            live_bytes = live.read_bytes()
            check("post-rotation write landed in live file",
                  b"AFTER-ROTATION" in live_bytes,
                  f"live={live_bytes!r}")
            check("post-rotation write did NOT pollute .1 backup",
                  b"AFTER-ROTATION" not in
                  live.with_suffix(".out.1").read_bytes())
        finally:
            os.close(fd)


async def test_disabled_when_zero() -> None:
    print("[disabled by config]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "chat_team.out"
        live.write_bytes(b"x" * 5000)
        # max_bytes=0 → reaper disabled, no task spawned, no rotation.
        rot = OutFileRotator(live, max_bytes=0, backup_count=5,
                             check_interval_seconds=10.0)
        rot.start()
        # Even calling _rotate_now directly should be a no-op because the
        # background loop is never created. But to be safe we also verify
        # the live file is untouched.
        await asyncio.sleep(0.05)
        check("disabled reaper → no task spawned", rot._task is None)
        check("disabled reaper → file untouched",
              live.stat().st_size == 5000)
        await rot.stop()


async def test_no_op_when_file_missing() -> None:
    print("[no-op when file missing]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "does-not-exist.out"
        rot = OutFileRotator(live, max_bytes=100, backup_count=2,
                             check_interval_seconds=10.0)
        # _maybe_rotate must not raise on FileNotFoundError.
        rot._maybe_rotate()
        check("missing file → no exception, no backup created",
              not live.exists() and
              not live.with_suffix(".out.1").exists())


async def test_real_async_loop() -> None:
    """End-to-end: run the actual asyncio background task with a short
    check interval, write past the cap through an O_APPEND FD, and verify
    rotation happens automatically.
    """
    print("[async background task rotates live file]")
    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "chat_team.out"
        # Short check interval so the reaper wakes multiple times during
        # the test. Default would be 5 minutes — useless in a smoke.
        rot = OutFileRotator(
            live, max_bytes=128, backup_count=2,
            check_interval_seconds=0.1,  # 100 ms — fast for the test
        )
        rot.start()
        try:
            fd = _write_oappend(live, b"")  # open the daemon-style FD
            try:
                # Write a steady stream of lines past the 128-byte cap. The
                # reaper wakes every 0.1s; we write for 2s total so the
                # reaper has ~20 chances to fire.
                deadline = asyncio.get_running_loop().time() + 2.0
                while asyncio.get_running_loop().time() < deadline:
                    os.write(fd, b"async-line-of-output-padding-XX\n")  # ~32 bytes
                    await asyncio.sleep(0.02)
                # Give the reaper one last chance to fire after the writes
                # stop, so the live file ends up truncated.
                await asyncio.sleep(0.4)
                check("async loop produced a backup",
                      live.with_suffix(".out.1").exists(),
                      f"dir={os.listdir(live.parent)}")
                # Live file should be ≤ cap after the final sweep.
                check("live file roughly under cap",
                      live.stat().st_size < 256,
                      f"size={live.stat().st_size}")
            finally:
                os.close(fd)
        finally:
            await rot.stop()


async def main() -> None:
    print("=== smoke_out_rotator ===")
    await test_basic_rotation()
    await test_backup_chain_bounded()
    await test_oappend_fd_survives_truncation()
    await test_disabled_when_zero()
    await test_no_op_when_file_missing()
    await test_real_async_loop()
    print()
    if _failed:
        print(f"\033[31mFAIL\033[0m  {_failed} check(s) failed")
        sys.exit(1)
    print(f"\033[32mALL PASS\033[0m  ({_passed} checks)")


if __name__ == "__main__":
    asyncio.run(main())
