"""模型与运行配置。针对 RTX 4090（24GB）显存优化。"""

from pathlib import Path

# RTX 4090 = 24GB GDDR6X；以下为实测可稳定运行的档位
MODEL_REGISTRY: dict[str, dict] = {
    "30b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "label": "32B · 4bit NF4（4090 大模型上限）",
        "load_in_4bit": True,
        "dtype": "float16",
    },
    "20b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "label": "14B · 4bit NF4（长讲义推荐）",
        "load_in_4bit": True,
        "dtype": "bfloat16",
    },
    "10b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "7B · BF16 全精度（4090 最快档）",
        "load_in_4bit": False,
        "dtype": "bfloat16",
    },
}

DATA_DIR = Path(__file__).resolve().parent / "data"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
LECTURE_TXT_NAME: str | None = "subtitles-519105.txt"

# 4090 推理优化
ENABLE_TF32 = True
USE_FLASH_ATTENTION = True  # 见 requirements.txt；未安装时自动降级

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.3
CHUNK_CHARS = 10_000  # 4090 + 4bit 下略保守，避免 OOM
