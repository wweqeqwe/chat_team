"""Session-level shared notebook backed by a single Markdown file.

Format: ``## key`` blocks. Multi-line values are preserved verbatim.
A sidecar ``notebook.index.json`` tracks updated_at per key.

The 4KB soft cap is enforced on write — if exceeded, ``write`` raises
``NotebookFull`` so the agent (LLM) gets a tool error and consolidates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_BLOCK_RE = re.compile(r"(?m)^##[ \t]+(?P<key>\S.*?)\s*$")


class NotebookFull(Exception):
    pass


class Notebook:
    def __init__(self, path: Path, max_bytes: int = 4096):
        self.path = path
        self.index_path = path.with_suffix(".index.json")
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- low-level parsing -------------------------------------------------

    def _load_blocks(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        text = self.path.read_text(encoding="utf-8")
        blocks: dict[str, str] = {}
        matches = list(_BLOCK_RE.finditer(text))
        for i, m in enumerate(matches):
            key = m.group("key").strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start:end].strip("\n")
            blocks[key] = value
        return blocks

    def _serialize(self, blocks: dict[str, str]) -> str:
        if not blocks:
            return ""
        parts = []
        for key, value in blocks.items():
            parts.append(f"## {key}\n{value}".rstrip() + "\n")
        return "\n".join(parts).rstrip() + "\n"

    def _atomic_write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(self.path.parent),
            prefix=".notebook.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        os.replace(tmp_path, self.path)

    def _load_index(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return {k: str(v) for k, v in data.items() if isinstance(k, str)}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self, index: dict[str, str]) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(self.index_path.parent),
            prefix=".notebook.index.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(json.dumps(index, ensure_ascii=False, indent=2))
            tmp_path = tmp.name
        os.replace(tmp_path, self.index_path)

    # ---- public API --------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self._load_blocks().keys())

    def read(self, key: str) -> str | None:
        return self._load_blocks().get(key)

    def dump(self) -> dict[str, str]:
        return self._load_blocks()

    def write(self, key: str, value: str) -> None:
        blocks = self._load_blocks()
        blocks[key] = value
        text = self._serialize(blocks)
        if len(text.encode("utf-8")) > self.max_bytes:
            raise NotebookFull(
                f"notebook would exceed {self.max_bytes} bytes; consolidate existing entries"
            )
        self._atomic_write(text)
        index = self._load_index()
        index[key] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_index(index)

    def delete(self, key: str) -> bool:
        blocks = self._load_blocks()
        if key not in blocks:
            return False
        del blocks[key]
        self._atomic_write(self._serialize(blocks))
        index = self._load_index()
        index.pop(key, None)
        self._save_index(index)
        return True

    def toc(self) -> str:
        """One-line summary of available keys + last-modified date."""
        index = self._load_index()
        keys = self.keys()
        if not keys:
            return "(empty)"
        return ", ".join(f"{k}({index.get(k, '?')})" for k in keys)

    def revision(self) -> str:
        """Stable digest of the notebook content and its TOC metadata.

        Agents use this to notice cross-role writes without rebuilding their
        leading system prompt on every LLM tool loop.  The notebook is capped
        at a few KB, so hashing both files is cheaper and more reliable than
        timestamp polling (same-day overwrites keep the TOC date unchanged).
        """
        digest = hashlib.blake2s(digest_size=16)
        for path in (self.path, self.index_path):
            try:
                payload = path.read_bytes()
            except FileNotFoundError:
                payload = b""
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._load_blocks().items())
