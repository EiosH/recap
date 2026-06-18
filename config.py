"""模型与运行配置。针对 RTX 4090（24GB）显存优化。"""

from pathlib import Path

# 三档 Qwen2.5 官方权重 + bitsandbytes INT4（NF4），Windows / Linux 通用。
# 若改用 *-AWQ 权重且已安装 gptqmodel，model_loader 会自动走 AWQ 路径。
MODEL_REGISTRY: dict[str, dict] = {
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
