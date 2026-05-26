"""讲义预处理：合并碎字幕行、去掉开头噪声，便于模型理解。"""

from __future__ import annotations

import re

_LECTURE_START = re.compile(
    r"(all right, we'?ll get started|welcome back|let'?s get started|today we)",
    re.I,
)

_NOISE_LINE = re.compile(
    r"^(i love you|i like the fabric|i want to show you how to use the fabric|i can see the fabric)\.?$",
    re.I,
)

def _drop_noise_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not _NOISE_LINE.match(ln.strip())]


def _find_start_index(lines: list[str]) -> int:
    for i, ln in enumerate(lines):
        if _LECTURE_START.search(ln):
            return i
    return 0


def _lines_to_paragraphs(lines: list[str], lines_per_para: int = 6) -> list[str]:
    """每若干条字幕行合并成一段（VTT 常无句号，不能仅靠句号切分）。"""
    paragraphs: list[str] = []
    buf: list[str] = []
    for ln in lines:
        buf.append(ln)
        if len(buf) >= lines_per_para:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


def prepare_lecture_text(raw: str) -> str:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    lines = _drop_noise_lines(lines)
    start = _find_start_index(lines)
    lines = lines[start:]
    paragraphs = _lines_to_paragraphs(lines, lines_per_para=6)
    return "\n\n".join(paragraphs).strip()
