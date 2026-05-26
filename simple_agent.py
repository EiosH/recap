#!/usr/bin/env python3
"""
LangChain + Hugging Face：读取 data 目录讲义 txt，生成课程知识提纲。
针对 RTX 4090 (24GB) 优化；每次运行生成完整 report。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from config import (
    CHUNK_CHARS,
    DATA_DIR,
    ENABLE_TF32,
    LECTURE_TXT_NAME,
    MAX_NEW_TOKENS,
    MODEL_REGISTRY,
    REPORT_DIR,
    TEMPERATURE,
    USE_FLASH_ATTENTION,
)
from extract_subtitles import extract_file
from gpu_monitor import (
    GpuSampler,
    RunMetrics,
    collect_gpu_info_text,
    cuda_mem_mb,
    cuda_peak_mb,
    print_gpu_info,
    print_run_metrics,
    reset_cuda_peak,
)
from report import RunReport, write_run_report

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
except ImportError:
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)


TEACHING_PROMPT = """你是一个助教。请把老师的讲课内容精炼总结成课程知识提纲。
要求：
- 不要赘述过多信息，条理清晰即可；
- 必须覆盖讲课中出现的全部知识点，不要遗漏重要概念。

讲课内容：
{lecture}"""


MERGE_PROMPT = """你是一个助教。下面是一堂课各片段的知识提纲，请合并为一份完整、不重复的课程知识提纲。
要求：简洁、结构化，确保包含所有片段中的知识点。

各片段提纲：
{partials}"""


def setup_4090() -> None:
    """RTX 4090 (Ada) 常用推理加速。"""
    if not torch.cuda.is_available():
        return
    if ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


def _resolve_dtype(name: str):
    return torch.bfloat16 if name == "bfloat16" else torch.float16


class TokenUsageHandler(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if not response.llm_output:
            return
        usage = response.llm_output.get("token_usage") or response.llm_output.get(
            "usage"
        )
        if not usage:
            return
        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        self.completion_tokens += usage.get("completion_tokens", 0) or 0
        self.total_tokens += usage.get("total_tokens", 0) or (
            self.prompt_tokens + self.completion_tokens
        )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def resolve_lecture_txt() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if LECTURE_TXT_NAME:
        txt_path = DATA_DIR / LECTURE_TXT_NAME
        if not txt_path.exists():
            vtt = txt_path.with_suffix(".vtt")
            if vtt.exists():
                print(f"未找到 {txt_path.name}，正在从 {vtt.name} 提取纯文本 ...")
                extract_file(vtt)
            else:
                raise FileNotFoundError(f"找不到讲义: {txt_path}")
        return txt_path

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if len(txt_files) == 1:
        return txt_files[0]
    if len(txt_files) == 0:
        vtt_files = list(DATA_DIR.glob("*.vtt"))
        if len(vtt_files) == 1:
            print(f"从 {vtt_files[0].name} 提取纯文本 ...")
            return extract_file(vtt_files[0])
        raise FileNotFoundError(f"{DATA_DIR} 中无 .txt 文件，请先运行 extract_subtitles.py")
    raise FileNotFoundError(
        f"{DATA_DIR} 中有多个 txt，请在 config.py 设置 LECTURE_TXT_NAME"
    )


def load_lecture_text() -> tuple[Path, str]:
    path = resolve_lecture_txt()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"讲义为空: {path}")
    return path, text


def chunk_lecture(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in text.split("\n"):
        add = len(para) + 1
        if buf and buf_len + add > size:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def build_llm(model_key: str) -> tuple[HuggingFacePipeline, float, str]:
    """加载模型，返回 (llm, 加载秒数, 量化描述)。"""
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg["model_id"]
    use_4bit = cfg["load_in_4bit"]
    dtype = _resolve_dtype(cfg["dtype"])

    setup_4090()
    print(f"\n正在加载 [4090] {model_key}: {model_id}")
    print(f"  {cfg['label']}")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": dtype,
    }

    quant_desc = f"{cfg['dtype']} 全精度"
    if use_4bit and torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs.pop("torch_dtype", None)
            quant_desc = "4bit NF4 (bitsandbytes，4090 推荐)"
        except ImportError:
            print("  警告: 未安装 bitsandbytes，回退全精度（可能 OOM）")

    if USE_FLASH_ATTENTION and torch.cuda.is_available():
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
        do_sample=True,
        return_full_text=False,
    )
    load_sec = time.perf_counter() - t0
    print(f"模型加载完成，耗时 {load_sec:.1f} 秒  [{quant_desc}]\n")
    return HuggingFacePipeline(pipeline=pipe), load_sec, quant_desc


def invoke_llm(
    llm: HuggingFacePipeline,
    prompt_text: str,
    token_handler: TokenUsageHandler,
) -> str:
    out = llm.invoke(prompt_text, config={"callbacks": [token_handler]})
    if isinstance(out, str):
        return out.strip()
    return str(out).strip()


def summarize_lecture(
    llm: HuggingFacePipeline,
    lecture: str,
    model_key: str,
    model_id: str,
    source_name: str,
) -> tuple[RunMetrics, int]:
    reset_cuda_peak()
    mem_before = cuda_mem_mb()
    token_handler = TokenUsageHandler()
    chunks = chunk_lecture(lecture, CHUNK_CHARS)

    teach_tpl = PromptTemplate.from_template(TEACHING_PROMPT)
    merge_tpl = PromptTemplate.from_template(MERGE_PROMPT)

    start = time.perf_counter()
    with GpuSampler(interval=0.5) as sampler:
        if len(chunks) == 1:
            prompt = teach_tpl.format(lecture=chunks[0])
            answer = invoke_llm(llm, prompt, token_handler)
        else:
            print(f"讲义较长，分 {len(chunks)} 段处理 ...")
            partials: list[str] = []
            for i, chunk in enumerate(chunks, 1):
                print(f"  处理片段 {i}/{len(chunks)} ...")
                header = f"【第 {i}/{len(chunks)} 段】\n"
                prompt = teach_tpl.format(lecture=header + chunk)
                partials.append(invoke_llm(llm, prompt, token_handler))
            merge_prompt = merge_tpl.format(partials="\n\n---\n\n".join(partials))
            answer = invoke_llm(llm, merge_prompt, token_handler)

    elapsed = time.perf_counter() - start

    prompt_t = token_handler.prompt_tokens
    completion_t = token_handler.completion_tokens
    total_t = token_handler.total_tokens
    if total_t == 0:
        prompt_t = _estimate_tokens(lecture)
        completion_t = _estimate_tokens(answer)
        total_t = prompt_t + completion_t

    metrics = RunMetrics(
        model_key=model_key,
        model_id=model_id,
        question=f"[讲义] {source_name} ({len(lecture):,} 字符, {len(chunks)} 段)",
        answer=answer,
        elapsed_sec=elapsed,
        prompt_tokens=prompt_t,
        completion_tokens=completion_t,
        total_tokens=total_t,
        gpu_mem_before_mb=mem_before,
        gpu_mem_after_mb=cuda_mem_mb(),
        gpu_mem_peak_mb=cuda_peak_mb(),
        gpu_util_samples=list(sampler.samples),
    )
    return metrics, len(chunks)


def print_menu() -> None:
    print("\n可用模型档位 (RTX 4090 24GB):")
    for k, cfg in MODEL_REGISTRY.items():
        print(f"  [{k}]  {cfg['model_id']}")
        print(f"        {cfg['label']}")
    print("\n命令: run 生成提纲 | 30b/20b/10b 切换模型 | gpu | quit")


def main() -> None:
    print_gpu_info()

    try:
        lecture_path, lecture_text = load_lecture_text()
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        sys.exit(1)

    print(f"\n已加载讲义: {lecture_path.name}")
    print(f"  字符数: {len(lecture_text):,}  行数: {len(lecture_text.splitlines()):,}")
    print_menu()

    current_key = "10b"
    llm: HuggingFacePipeline | None = None
    loaded_key: str | None = None
    load_time_sec = 0.0
    quant_mode = ""

    while True:
        cfg = MODEL_REGISTRY[current_key]
        print(f"\n当前模型: {current_key}  ({cfg['model_id']})")
        raw = input("命令 > ").strip().lower()

        if not raw:
            continue
        if raw in ("quit", "exit", "q"):
            print("再见。")
            break
        if raw == "gpu":
            print_gpu_info()
            continue
        if raw in MODEL_REGISTRY:
            current_key = raw
            loaded_key = None
            llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"已切换到 {current_key}，输入 run 开始生成提纲。")
            continue
        if raw != "run":
            print("未知命令。可用: run | 30b/20b/10b | gpu | quit")
            continue

        model_id = cfg["model_id"]
        if loaded_key != current_key:
            if llm is not None and torch.cuda.is_available():
                del llm
                torch.cuda.empty_cache()
            llm, load_time_sec, quant_mode = build_llm(current_key)
            loaded_key = current_key

        assert llm is not None
        started_at = datetime.now()
        gpu_snapshot = collect_gpu_info_text()

        print(f"\n>>> 正在用 [{current_key}] 生成课程知识提纲 ...\n")
        metrics, num_chunks = summarize_lecture(
            llm,
            lecture_text,
            current_key,
            model_id,
            lecture_path.name,
        )
        finished_at = datetime.now()

        print_run_metrics(metrics)

        outline_path = lecture_path.with_name(lecture_path.stem + "-outline.txt")
        outline_path.write_text(metrics.answer, encoding="utf-8")
        print(f"提纲已保存: {outline_path}")

        report = RunReport(
            metrics=metrics,
            started_at=started_at,
            finished_at=finished_at,
            lecture_path=lecture_path.resolve(),
            lecture_chars=len(lecture_text),
            lecture_lines=len(lecture_text.splitlines()),
            num_chunks=num_chunks,
            outline_path=outline_path.resolve(),
            load_time_sec=load_time_sec,
            quant_mode=quant_mode,
            gpu_snapshot=gpu_snapshot,
            extra={
                "chunk_chars": CHUNK_CHARS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            },
        )
        report_path = write_run_report(REPORT_DIR, report)
        print(f"运行报告已保存: {report_path}\n")


if __name__ == "__main__":
    main()
