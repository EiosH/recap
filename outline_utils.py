"""提纲文件命名与查找（按模型区分）。"""

from __future__ import annotations

import re
from pathlib import Path

OUTLINE_SUFFIX_RE = re.compile(r"^(.+)-outline-(\d+b|\w+)$")


def outline_path(data_dir: Path, lecture_stem: str, model_key: str) -> Path:
    """例: subtitles-519105-outline-10b.txt"""
    return data_dir / f"{lecture_stem}-outline-{model_key}.txt"


def parse_outline_path(path: Path) -> tuple[str, str] | None:
    """从文件名解析 (lecture_stem, model_key)。"""
    m = OUTLINE_SUFFIX_RE.match(path.stem)
    if not m:
        return None
    return m.group(1), m.group(2)


def list_outlines(data_dir: Path, runs_dir: Path | None = None) -> list[Path]:
    """列出所有提纲文件，按修改时间倒序。"""
    candidates: list[Path] = []

    if data_dir.exists():
        candidates.extend(data_dir.glob("*-outline-*.txt"))
        # 兼容旧命名
        for p in data_dir.glob("*-outline.txt"):
            if p not in candidates:
                candidates.append(p)

    if runs_dir and runs_dir.exists():
        for run in runs_dir.iterdir():
            if not run.is_dir():
                continue
            for name in ("assembled-outline.txt", "outline-*.txt"):
                candidates.extend(run.glob(name))

    # 去重，按 mtime 倒序
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            unique.append(p)
    return unique


def latest_outline(data_dir: Path, runs_dir: Path | None = None) -> Path | None:
    outlines = list_outlines(data_dir, runs_dir)
    return outlines[0] if outlines else None


def outline_label(path: Path) -> str:
    """人类可读标签。"""
    parsed = parse_outline_path(path)
    if parsed:
        stem, model = parsed
        return f"{stem} [{model}]"
    if "runs" in path.parts:
        return f"{path.parent.name}/assembled-outline"
    return path.name
