"""模型与运行配置。针对 RTX 4090（24GB）显存优化。"""

from pathlib import Path

# RTX 4090 = 24GB GDDR6X。
# 30b / 20b 使用真实 Qwen2.5 权重，INT4（bitsandbytes NF4）加载。
# 注意：32B 即使 4bit 仍约需 18–20GB 显存；未安装 bitsandbytes 时 simple_agent 会降级。
MODEL_REGISTRY: dict[str, dict] = {
    "32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "label": "32B · AWQ 4bit（高质量档，4090接近上限，适合复杂推理）",
        "quant": "awq",
        "dtype": "float16",
        "role": "fallback_high_quality",
    },
    "14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "label": "14B · AWQ 4bit（主力档：RAG + 多学生并发最优平衡）",
        "quant": "awq",
        "dtype": "float16",

        # ⭐关键（你系统真正要关心的）
        "role": "production_default",

        # 建议推理参数（给你直接能用）
        "generation_config": {
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9
        }
    },
     "8b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "label": "7B · AWQ 4bit（高并发轻量档，用于峰值流量）",
        "quant": "awq",
        "dtype": "float16",
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
