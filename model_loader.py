"""按 MODEL_REGISTRY 加载 HuggingFace 模型（AWQ / bitsandbytes INT4 / 全精度）。"""

from __future__ import annotations

import time

from config import (
    DO_SAMPLE,
    ENABLE_TF32,
    MAX_NEW_TOKENS,
    MODEL_REGISTRY,
    TEMPERATURE,
    USE_FLASH_ATTENTION,
)

try:
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
except ImportError as exc:
    raise ImportError("请先安装依赖: pip install -r requirements.txt") from exc


def _resolve_dtype(name: str):
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _setup_cuda() -> None:
    if not torch.cuda.is_available():
        return
    if ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


def _ensure_gptqmodel() -> None:
    try:
        import gptqmodel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "加载 AWQ 模型需要 gptqmodel。请执行:\n"
            "  pip install gptqmodel\n"
            "若安装失败可尝试:\n"
            "  pip install gptqmodel --no-build-isolation"
        ) from exc


def build_llm(model_key: str) -> tuple[HuggingFacePipeline, float, str]:
    """加载模型，返回 (llm, 加载秒数, 量化描述)。"""
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg["model_id"]
    dtype = _resolve_dtype(cfg["dtype"])
    quant = cfg.get("quant", "")

    _setup_cuda()
    print(f"\n正在加载 [{model_key}]: {model_id}")
    print(f"  {cfg['label']}")
    if cfg.get("role"):
        print(f"  角色: {cfg['role']}")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "cuda:0" if torch.cuda.is_available() else "auto",
    }

    if quant == "awq":
        _ensure_gptqmodel()
        model_kwargs["torch_dtype"] = dtype
        quant_desc = "AWQ INT4 (gptqmodel)"
    elif cfg.get("load_in_4bit") and torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            quant_desc = "INT4 NF4 (bitsandbytes)"
        except ImportError:
            print("  警告: 未安装 bitsandbytes，回退全精度加载。")
            model_kwargs["torch_dtype"] = dtype
            quant_desc = f"{cfg['dtype']} 全精度"
    else:
        model_kwargs["torch_dtype"] = dtype
        quant_desc = f"{cfg['dtype']} 全精度"

    if USE_FLASH_ATTENTION and torch.cuda.is_available() and quant != "awq":
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  已启用 Flash Attention 2")
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        return_full_text=False,
    )
    load_sec = time.perf_counter() - t0
    print(f"模型加载完成，耗时 {load_sec:.1f} 秒  [{quant_desc}]\n")

    hf_llm = HuggingFacePipeline(pipeline=pipe)
    hf_llm.pipeline.tokenizer = tokenizer
    return hf_llm, load_sec, quant_desc
