#!/usr/bin/env python3
"""
从 data 目录中的 WebVTT (.vtt) 字幕提取纯文本，去掉时间轴与 cue 编号，输出同名 .txt。

识别格式: WebVTT（含 00:00:00.000 --> 00:00:00.000 时间行）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# WebVTT 时间轴行
TIMESTAMP_LINE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}\.\d{3}\s*$"
)
# 行内夹杂的时间轴（如损坏的 WEBVTT 首行）
INLINE_TIMESTAMP = re.compile(
    r"\d{1,2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}\.\d{3}"
)
CUE_ID = re.compile(r"^\d+$")
SKIP_PREFIXES = ("WEBVTT", "NOTE", "STYLE", "REGION")


def is_metadata_line(line: str) -> bool:
    upper = line.strip().upper()
    if not upper:
        return True
    if upper.startswith(SKIP_PREFIXES):
        return True
    if TIMESTAMP_LINE.match(line):
        return True
    if INLINE_TIMESTAMP.search(line):
        return True
    if CUE_ID.match(line.strip()):
        return True
    return False


def vtt_to_plain_text(vtt_path: Path) -> str:
    raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    paragraphs: list[str] = []
    prev: str | None = None

    for line in raw.splitlines():
        text = line.strip()
        if is_metadata_line(text):
            continue
        if not text:
            continue
        if text == prev:
            continue
        paragraphs.append(text)
        prev = text

    return "\n".join(paragraphs)


def extract_file(vtt_path: Path) -> Path:
    out_path = vtt_path.with_suffix(".txt")
    plain = vtt_to_plain_text(vtt_path)
    out_path.write_text(plain, encoding="utf-8")
    return out_path


def main() -> None:
    targets = list(DATA_DIR.glob("*.vtt"))
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]

    if not targets:
        print(f"未在 {DATA_DIR} 找到 .vtt 文件")
        sys.exit(1)

    for vtt in targets:
        if not vtt.exists():
            print(f"跳过（不存在）: {vtt}")
            continue
        out = extract_file(vtt)
        chars = out.read_text(encoding="utf-8")
        print(f"已识别: WebVTT 字幕")
        print(f"  输入: {vtt.name} ({vtt.stat().st_size:,} 字节)")
        print(f"  输出: {out.name} ({len(chars):,} 字符, {len(chars.splitlines()):,} 行)")


if __name__ == "__main__":
    main()
