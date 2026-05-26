"""分段总结落盘：每段写完即存文件，最终从磁盘拼接，避免内存堆积与合并丢内容。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ChunkRun:
    run_dir: Path
    model_key: str
    source_name: str

    @classmethod
    def create(cls, base_dir: Path, model_key: str, source_name: str) -> "ChunkRun":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / f"{stamp}-{model_key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = run_dir / "meta.txt"
        meta.write_text(
            f"source={source_name}\nmodel_key={model_key}\n",
            encoding="utf-8",
        )
        return cls(run_dir=run_dir, model_key=model_key, source_name=source_name)

    def summary_path(self, index: int) -> Path:
        return self.run_dir / f"summary-{index:03d}.txt"

    def save_summary(self, index: int, text: str) -> Path:
        path = self.summary_path(index)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    def list_summaries(self) -> list[Path]:
        return sorted(self.run_dir.glob("summary-*.txt"))

    def assemble_concat(self) -> str:
        """从磁盘读取各段总结，直接拼接（不丢内容）。"""
        parts: list[str] = []
        files = self.list_summaries()
        for i, path in enumerate(files, 1):
            body = path.read_text(encoding="utf-8").strip()
            if not body:
                continue
            parts.append(f"## 第 {i} 段\n\n{body}")
        return "\n\n".join(parts).strip()

    def save_assembled(self, text: str) -> Path:
        out = self.run_dir / "assembled-outline.txt"
        out.write_text(text.strip() + "\n", encoding="utf-8")
        return out
