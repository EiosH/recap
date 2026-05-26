"""模型与运行配置。针对 RTX 4090（24GB）显存优化。"""

from pathlib import Path

# RTX 4090 = 24GB GDDR6X。
# 注意：在 Windows / 未成功安装 bitsandbytes 时，大模型（32B）会非常容易 OOM，
# 所以这里把 30b / 20b / 10b 档位约束在 14B / 7B 以内，优先走 4bit。
MODEL_REGISTRY: dict[str, dict] = {
    # 尽量大但仍相对可控的档位（优先 4bit）
    "30b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "label": "14B · 4bit NF4（大模型档，需 4bit 或高显存）",
        "load_in_4bit": True,
        "dtype": "float16",
    },
    # 中档：7B 4bit，适合长讲义
    "20b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "7B · 4bit NF4（推荐默认档，兼顾速度与效果）",
        "load_in_4bit": True,
        "dtype": "bfloat16",
    },
    # 小档：7B BF16，全精度更快
    "10b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "7B · BF16 全精度（最稳妥，优先用于排错）",
        "load_in_4bit": False,
        "dtype": "bfloat16",
    },
}

DATA_DIR = Path(__file__).resolve().parent / "data"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
LECTURE_TXT_NAME: str | None = "subtitles-519105.txt"

# 4090 推理优化（在 Windows+4090 上同样生效）
ENABLE_TF32 = True
USE_FLASH_ATTENTION = True  # 见 requirements.txt；未安装时自动降级

# 为避免 OOM，适当降低生成长度与单段字符数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
CHUNK_CHARS = 4_000  # 单段约 4k 字符，减小注意力矩阵大小
