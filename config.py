"""模型与运行配置。针对 RTX 4090（24GB）显存优化。"""

from pathlib import Path

# 默认走本地 Ollama（与 course-db 同一个模型）；其余档位为 HuggingFace + bitsandbytes INT4。
# 若 HF 档位改用 *-AWQ 权重且已安装 gptqmodel，model_loader 会自动走 AWQ 路径。
DEFAULT_MODEL = "27b"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 600
# Ollama 下传为 options.num_ctx —— 不设时用默认值（约 2048~4096），
# 超出部分会从消息开头被静默截掉（system prompt 最先丢）。
OLLAMA_NUM_CTX = 16384

MODEL_REGISTRY: dict[str, dict] = {
    "27b": {
        "backend": "ollama",
        "model_id": "qwen3.8:27b",
        "label": "Ollama 本地 qwen3.8:27b（默认档，与 course-db 一致）",
        "quant": "ollama",
        "role": "default",
    },
    "32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "label": "32B · INT4 NF4（高质量档，4090 约 18–20GB 显存）",
        "load_in_4bit": True,
        "dtype": "float16",
        "role": "fallback_high_quality",
    },
    "14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "label": "14B · INT4 NF4（主力档：RAG + 多学生并发最优平衡）",
        "load_in_4bit": True,
        "dtype": "float16",
        "role": "production_default",
        "generation_config": {
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
        },
    },
    "8b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "7B · INT4 NF4（高并发轻量档，用于峰值流量）",
        "load_in_4bit": True,
        "dtype": "bfloat16",
        "role": "high_concurrency_fallback",
    },
}

DATA_DIR = Path(__file__).resolve().parent / "data"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
CHUNK_RUN_DIR = DATA_DIR / "runs"  # 每段总结临时落盘目录
LECTURE_TXT_NAME: str | None = "CSC447-lecture-2-subtitles.txt"

# 4090 推理优化（在 Windows+4090 上同样生效）
ENABLE_TF32 = True
USE_FLASH_ATTENTION = True  # 见 requirements.txt；未安装时自动降级

# 单段只出简短笔记（prompt 要求 ≤250 词），这里是硬上限
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.1
DO_SAMPLE = False  # 总结任务用贪心解码，减少胡编
CHUNK_CHARS = 4_000

# 按 token 控制每段输入上限
MAX_INPUT_TOKENS = 3072

# 最终提纲：一次读入全部分段笔记，按知识结构重组，目标 2000~3000 英文单词。
# 输入 + 输出要放得进 OLLAMA_NUM_CTX，否则 Ollama 会从开头静默截断。
MAX_OUTLINE_INPUT_TOKENS = 10000
MAX_NEW_TOKENS_OUTLINE = 5000

# 命令行 chat 对话
MAX_NEW_TOKENS_CHAT = 1024
MAX_OUTLINE_TOKENS_CHAT = 6000
CHAT_TEMPERATURE = 0.7
CHAT_DO_SAMPLE = True
