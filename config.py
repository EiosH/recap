"""模型与运行配置。可通过环境变量 HF_TOKEN 访问 gated 模型。"""

from pathlib import Path

# 约 30B / 20B / 10B 档位的 HuggingFace 模型 ID（可按本机显存改小一号）
MODEL_REGISTRY = {
    "30b": "Qwen/Qwen2.5-32B-Instruct",
    "20b": "google/gemma-2-27b-it",
    "10b": "Qwen/Qwen2.5-14B-Instruct",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
# 讲课 txt 文件名；为 None 时自动使用 data 下唯一 .txt
LECTURE_TXT_NAME: str | None = "subtitles-519105.txt"

# 4bit 量化可显著降低显存；显存充足可改为 load_in_4bit=False
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.3
# 长讲义分块字符数（避免超出上下文）
CHUNK_CHARS = 12_000
