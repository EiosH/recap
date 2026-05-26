"""每次运行结束后生成完整 report 文件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gpu_monitor import RunMetrics


@dataclass
class RunReport:
    metrics: RunMetrics
    started_at: datetime
    finished_at: datetime
    lecture_path: Path
    lecture_chars: int
    lecture_lines: int
    num_chunks: int
    outline_path: Path | None
    load_time_sec: float
    quant_mode: str
    gpu_snapshot: str = ""
    extra: dict = field(default_factory=dict)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def write_run_report(report_dir: Path, report: RunReport) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    m = report.metrics
    stamp = report.started_at.strftime("%Y%m%d-%H%M%S")
    filename = f"run-{stamp}-{m.model_key}.md"
    path = report_dir / filename

    lines = [
        "# 课程提纲生成 — 运行报告",
        "",
        "## 基本信息",
        f"- **开始时间**: {_fmt_dt(report.started_at)}",
        f"- **结束时间**: {_fmt_dt(report.finished_at)}",
        f"- **总耗时**: {m.elapsed_sec:.2f} 秒",
        f"- **模型档位**: {m.model_key}",
        f"- **HuggingFace 模型**: `{m.model_id}`",
        f"- **量化 / 精度**: {report.quant_mode}",
        f"- **模型加载耗时**: {report.load_time_sec:.2f} 秒",
        "",
        "## 讲义输入",
        f"- **文件**: `{report.lecture_path}`",
        f"- **字符数**: {report.lecture_chars:,}",
        f"- **行数**: {report.lecture_lines:,}",
        f"- **分块数**: {report.num_chunks}",
        f"- **分块大小 (CHUNK_CHARS)**: {report.extra.get('chunk_chars', 'N/A')}",
        "",
        "## 性能指标",
        f"- **显存运行前 (MB)**: {m.gpu_mem_before_mb:.0f}",
        f"- **显存运行后 (MB)**: {m.gpu_mem_after_mb:.0f}",
        f"- **显存峰值 (MB)**: {m.gpu_mem_peak_mb:.0f}",
        f"- **GPU 利用率平均**: {m.gpu_util_avg:.1f}%",
        f"- **GPU 利用率峰值**: {m.gpu_util_max:.1f}%",
        f"- **GPU 采样次数**: {len(m.gpu_util_samples)}",
        "",
        "## Token 消耗",
        f"- **输入 (prompt)**: {m.prompt_tokens:,}",
        f"- **输出 (completion)**: {m.completion_tokens:,}",
        f"- **合计**: {m.total_tokens:,}",
        "",
        "## 推理配置",
        f"- **MAX_NEW_TOKENS**: {report.extra.get('max_new_tokens', 'N/A')}",
        f"- **TEMPERATURE**: {report.extra.get('temperature', 'N/A')}",
        f"- **组装模式**: {report.extra.get('assemble_mode', 'N/A')}",
        f"- **分段目录**: `{report.extra.get('chunk_run_dir', 'N/A')}`",
        f"- **任务描述**: {m.question}",
        "",
        "## 输出文件",
    ]
    if report.outline_path:
        lines.append(f"- **课程提纲**: `{report.outline_path}`")
    lines.extend([
        f"- **本报告**: `{path}`",
        "",
        "## 显卡快照（运行前）",
        "",
        "```",
        report.gpu_snapshot.strip() or "(无)",
        "```",
        "",
        "## 生成的课程知识提纲",
        "",
        m.answer,
        "",
        "---",
        "*报告由 simple_agent.py 自动生成*",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
