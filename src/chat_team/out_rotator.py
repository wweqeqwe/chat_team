"""Copytruncate-style rotator for ``~/.chat_team/logs/chat_team.out``.

The daemon's stdout/stderr are redirected to ``chat_team.out`` via
``os.open(path, O_WRONLY|O_CREAT|O_APPEND)`` + ``os.dup2(.., 1/2)`` in
``chat_team.daemon.daemonize_and_run``. Because the file is held open at the
FD level (not via a Python logging handler), a ``RotatingFileHandler`` can't
intercept it — that's why ``chat_team.out`` grew without bound.

This module provides a copytruncate reaper that runs as an asyncio background
task. It periodically ``stat()``s the file; when the size exceeds the cap it:

1. Reads the current file contents.
2. Shifts ``.N`` → ``.N+1`` for ``N = backup_count-1 .. 1`` (deleting the
   oldest), writes the current contents into ``.1``.
3. ``os.truncate(0)``s the live file.

``os.truncate(0)`` is the key trick: the daemon's open FD was created with
``O_APPEND``, so every ``write(2)`` seeks to the *current* end-of-file
before writing. After truncation that's byte 0, so subsequent stdout/stderr
writes resume at the top of the same inode — no FD re-opening needed.

All filesystem ops run in ``asyncio.to_thread`` so a slow disk never blocks
the event loop. Failures are logged at WARNING and never propagate — a
rotation hiccup must not take down the daemon.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class OutFileRotator:
    """Background copytruncate rotator for ``chat_team.out``.

    One instance per daemon process; started via :meth:`start` from
    ``_async_main`` and stopped via :meth:`stop` on shutdown. The reaper
    loop is intentionally simple: ``stat`` → maybe rotate → sleep. The
    cap is approximate — between wakeups the file can grow past the
    threshold by ``check_interval × write_rate``, but the cap holds in the
    steady state.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int,
        backup_count: int,
        check_interval_seconds: float = 300.0,
    ) -> None:
        self._path = path
        self._max_bytes = max(0, int(max_bytes))
        self._backup_count = max(0, int(backup_count))
        # 1s floor: the reaper is just a stat() + (rarely) a copytruncate,
        # so checking every second is cheap; the floor only prevents a
        # misconfigured caller from spinning in a tight loop.
        self._interval = max(1.0, float(check_interval_seconds))
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        if self._max_bytes <= 0 or self._backup_count <= 0:
            # Disabled by config — log once so the maintainer knows the
            # reaper is intentionally off rather than broken.
            log.info(
                "out-file rotation disabled (max_bytes=%d backup_count=%d)",
                self._max_bytes, self._backup_count,
            )
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="out-rotator")

    async def stop(self, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        self._task = None

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.to_thread(self._maybe_rotate)
                except Exception as e:  # noqa: BLE001
                    log.warning("out-file rotation failed: %r", e)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    def _maybe_rotate(self) -> None:
        try:
            size = os.path.getsize(self._path)
        except FileNotFoundError:
            return
        if size <= self._max_bytes:
            return
        log.info(
            "rotating %s (size=%d > cap=%d, backups=%d)",
            self._path.name, size, self._max_bytes, self._backup_count,
        )
        self._rotate_now()

    def _rotate_now(self) -> None:
        """Perform one copytruncate rotation. Sync — call via to_thread."""
        # 1. Shift existing backups: .N → .N+1, dropping the oldest.
        # We have backups .1 .. .(backup_count-1) plus the live file.
        # After rotation we want: live → .1, .1 → .2, ..., .(N-1) → .N,
        # and the oldest (.N, was the (N+1)th backup) gets deleted.
        n = self._backup_count
        # Delete the would-be-oldest backup so the subsequent chain of
        # os.rename() calls never lands on a pre-existing inode.
        oldest = self._path.with_suffix(self._path.suffix + f".{n}")
        if oldest.exists():
            try:
                oldest.unlink()
            except OSError as e:
                log.warning("could not delete oldest backup %s: %r", oldest, e)
                # Keep going — a rename target that exists is replaced on
                # POSIX anyway, but being explicit avoids surprising the
                # maintainer with a sudden rename-failure on Windows.

        for i in range(n - 1, 0, -1):
            src = self._path.with_suffix(self._path.suffix + f".{i}")
            dst = self._path.with_suffix(self._path.suffix + f".{i + 1}")
            if src.exists():
                try:
                    os.replace(src, dst)
                except OSError as e:
                    log.warning("backup shift %s → %s failed: %r", src, dst, e)

        # 2. Copy the live file into .1 (copy, not rename — the live FD
        #    must keep pointing at the original inode).
        backup1 = self._path.with_suffix(self._path.suffix + ".1")
        try:
            shutil.copy2(self._path, backup1)
        except OSError as e:
            log.warning("could not copy %s → %s: %r", self._path, backup1, e)
            return

        # 3. Truncate the live file. The open O_APPEND FD will resume writing
        #    at byte 0 on the next write — see module docstring.
        try:
            os.truncate(self._path, 0)
        except OSError as e:
            log.warning("could not truncate %s after backup: %r", self._path, e)
            # The backup is already on disk; the worst case is the live file
            # keeps growing until the next sweep truncates it. Don't delete
            # the backup we just wrote.
