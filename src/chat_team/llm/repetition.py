"""Degenerate-repetition detection for streaming model output.

When a model "loops" it collapses into emitting the exact same block of
text over and over until it hits its output cap.  Waiting for the cap
wastes tokens and wall-clock time; instead we watch the accumulated
output while it streams and abort as soon as the tail becomes many
verbatim copies of one block.  The caller (OpenAIChatCompletionProvider)
treats the abort as a retryable error — a fresh sample almost always
breaks the loop.

Detection is exact-match only (conservative: merely *similar* lines never
trip it) and period-agnostic: the minimal period of a tail window comes
from the KMP prefix function, so the repeating unit can be anything from a
short phrase ("好的，我来处理。") to a whole paragraph, at any alignment.
"""
from __future__ import annotations

# Tail window sizes (chars), checked smallest-first so a late-onset loop
# is caught as early as possible.  A window only "hits" when it lies
# entirely inside the looping region, hence several sizes.
_WINDOWS: tuple[int, ...] = (128, 192, 256, 512, 1024, 2048)

# How many new characters must accumulate between checks (bounds CPU use;
# degenerate loops emit thousands of characters, so this still aborts far
# earlier than any output cap).
CHECK_STEP_CHARS = 256


def _min_copies(period: int) -> int:
    """Shorter repeating blocks need more copies to count as degenerate
    (a table or list may legitimately repeat a short line a few times);
    long blocks need fewer."""
    if period <= 16:
        return 8
    if period <= 32:
        return 5
    return 3


def _prefix_function(s: str) -> list[int]:
    pi = [0] * len(s)
    for i in range(1, len(s)):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def find_degenerate_repeat(text: str) -> str | None:
    """Return a short sample of the repeating block if ``text``'s tail is
    degenerate repetition, else ``None``.
    """
    n = len(text)
    if n < _WINDOWS[0]:
        return None
    for window in _WINDOWS:
        if n < window:
            break
        tail = text[-window:]
        pi = _prefix_function(tail)
        period = window - pi[-1]
        if period >= window:
            continue                       # aperiodic tail
        if window // period >= _min_copies(period):
            return tail[:min(period, 40)]
    return None
