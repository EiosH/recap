"""分段总结落盘：每段写完即存文件，支持中断后从已完成的段继续，全部完成后清理。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkRun:
    run_dir: Path
    model_key: str
    source_name: str

    @classmethod
    def open(cls, base_dir: Path, model_key: str, source_name: str) -> "ChunkRun":
        """同一讲义 + 同一模型固定使用一个目录，中断后再次运行会复用它。"""
        run_dir = base_dir / f"{Path(source_name).stem}-{model_key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.txt").write_text(
            f"source={source_name}\nmodel_key={model_key}\n",
            encoding="utf-8",
        )
        return cls(run_dir=run_dir, model_key=model_key, source_name=source_name)

    def summary_path(self, index: int) -> Path:
        return self.run_dir / f"summary-{index:03d}.txt"

    def input_path(self, index: int) -> Path:
        return self.run_dir / f"input-{index:03d}.txt"

    def is_done(self, index: int, text: str) -> bool:
        """该段已有总结，且当时的输入与本次切分结果一致。"""
        inp = self.input_path(index)
        return (
            self.summary_path(index).exists()
            and inp.exists()
            and inp.read_text(encoding="utf-8") == text.strip() + "\n"
        )

    def save_chunk(self, index: int, text: str, summary: str) -> Path:
        # input 最后写：只有 summary 写完后它才与本次输入一致，is_done 才会成立
        self.summary_path(index).write_text(summary.strip() + "\n", encoding="utf-8")
        self.input_path(index).write_text(text.strip() + "\n", encoding="utf-8")
        return self.summary_path(index)

    def list_summaries(self, count: int) -> list[Path]:
        return [self.summary_path(i) for i in range(1, count + 1)]

    def assemble_concat(self, count: int) -> str:
        """从磁盘读取各段总结，直接拼接（不丢内容）。"""
        parts: list[str] = []
        for i, path in enumerate(self.list_summaries(count), 1):
            body = path.read_text(encoding="utf-8").strip()
            if body:
                parts.append(f"## Part {i}\n\n{body}")
        return "\n\n".join(parts).strip()

    def cleanup(self) -> None:
        shutil.rmtree(self.run_dir, ignore_errors=True)
