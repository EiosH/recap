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
CHUNK_RUN_DIR = DATA_DIR / "runs"  # 每段总结临时落盘目录
LECTURE_TXT_NAME: str | None = "subtitles-519105.txt"

# 4090 推理优化（在 Windows+4090 上同样生效）
ENABLE_TF32 = True
USE_FLASH_ATTENTION = True  # 见 requirements.txt；未安装时自动降级

# 单段总结允许更长输出，避免提纲被截断
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.1
DO_SAMPLE = False  # 总结任务用贪心解码，减少胡编
CHUNK_CHARS = 4_000

# 按 token 控制每段输入上限
MAX_INPUT_TOKENS = 3072

# 最终组装方式：
# - concat：从磁盘读取各段 summary 直接拼接（推荐，不丢知识点）
# - merge：再用 LLM 合并（更短，可能丢细节）
ASSEMBLE_MODE = "concat"
MERGE_BATCH = 2
MAX_NEW_TOKENS_MERGE = 1024

# 命令行 chat 对话
MAX_NEW_TOKENS_CHAT = 1024
MAX_OUTLINE_TOKENS_CHAT = 6000
CHAT_TEMPERATURE = 0.7
CHAT_DO_SAMPLE = True
