#!/usr/bin/env python3
"""
LangChain + Hugging Face：读取 data 目录讲义 txt，生成课程知识提纲。
针对 RTX 4090 (24GB) 优化；每次运行生成完整 report。
"""

from __future__ import annotations

import sys
import time
import os
import gc
import re
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from config import (
    ASSEMBLE_MODE,
    CHUNK_CHARS,
    CHUNK_RUN_DIR,
    DATA_DIR,
    DO_SAMPLE,
    ENABLE_TF32,
    LECTURE_TXT_NAME,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_MERGE,
    MERGE_BATCH,
    MODEL_REGISTRY,
    REPORT_DIR,
    TEMPERATURE,
    USE_FLASH_ATTENTION,
)
from chunk_store import ChunkRun
from lecture_prep import prepare_lecture_text
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

# 减少 CUDA 内存碎片（对“跑到最后 merge 才 OOM”的情况很关键）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SYSTEM_PROMPT = """你是大学课程助教。你的任务是根据课堂录音自动转写的英文字幕，提炼课程知识提纲（用中文输出）。

严格要求：
1. 只总结字幕里实际讲到的内容，不要编造、不要臆测；
2. 忽略口语重复、语气词、与课程无关的闲聊；
3. 保留专业术语（必要时中英对照）；
4. 用清晰的条目列出知识点，确保覆盖该段所有重要概念。"""


USER_CHUNK_TEMPLATE = """以下是一段课堂录音的英文字幕（自动转写，可能有识别错误）：

---
{lecture}
---

请提炼本段课程知识提纲（中文，条目式）。"""


MERGE_USER_TEMPLATE = """以下是同一堂课各片段的知识提纲，请合并为一份完整提纲（中文）：
- 结构化、去重，但不要删减知识点；
- 保留术语与概念。

各片段提纲：
---
{partials}
---"""


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
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"讲义为空: {path}")
    text = prepare_lecture_text(raw)
    if not text:
        raise ValueError(f"讲义预处理后为空: {path}")
    return path, text


def format_chat_prompt(tokenizer, user_content: str) -> str:
    """Qwen Instruct 必须用 chat template，否则容易答非所问。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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
        # 单卡 4090：显式放到 cuda:0，避免 auto 造成奇怪的 offload/切分行为
        "device_map": "cuda:0" if torch.cuda.is_available() else "auto",
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
            quant_desc = "4bit NF4 (bitsandbytes，推荐)"
        except ImportError:
            # 在 Windows / 未安装 bitsandbytes 的环境下，强行全精度加载大模型极易 OOM。
            # 这里直接降级到 7B BF16，以保证一定能跑通。
            print("  警告: 未安装 bitsandbytes，本次将自动降级为 7B BF16 以避免 OOM。")
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            use_4bit = False
            dtype = torch.bfloat16
            model_kwargs["torch_dtype"] = dtype

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
        do_sample=DO_SAMPLE,
        return_full_text=False,
    )
    load_sec = time.perf_counter() - t0
    print(f"模型加载完成，耗时 {load_sec:.1f} 秒  [{quant_desc}]\n")
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    hf_llm.pipeline.tokenizer = tokenizer  # 确保后续能取到 tokenizer
    return hf_llm, load_sec, quant_desc


def _token_len(tokenizer, text: str) -> int:
    return int(len(tokenizer.encode(text, add_special_tokens=False)))


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def chunk_lecture_tokens(tokenizer, text: str, max_tokens: int) -> list[str]:
    """按 token 切分讲义（以段落为单位），保证每段不超过 max_tokens。"""
    # 先按空行分段（prepare_lecture_text 已合并成段落）
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for para in paragraphs:
        para_tokens = _token_len(tokenizer, para) + 2
        if buf and buf_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_tokens = 0
        if para_tokens > max_tokens:
            # 单段过长，按句子再切
            for sent in re.split(r"(?<=[.?!])\s+", para):
                sent = sent.strip()
                if not sent:
                    continue
                st = _token_len(tokenizer, sent) + 1
                if buf and buf_tokens + st > max_tokens:
                    chunks.append("\n\n".join(buf))
                    buf = []
                    buf_tokens = 0
                buf.append(sent)
                buf_tokens += st
        else:
            buf.append(para)
            buf_tokens += para_tokens

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def invoke_llm(
    llm: HuggingFacePipeline,
    tokenizer,
    user_content: str,
    token_handler: TokenUsageHandler,
    max_new_tokens: int | None = None,
) -> str:
    prompt_text = format_chat_prompt(tokenizer, user_content)
    pipe = llm.pipeline
    kwargs: dict = {}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        raw = pipe(prompt_text, **kwargs)
        if isinstance(raw, list) and raw:
            item = raw[0]
            out = item.get("generated_text", str(item)) if isinstance(item, dict) else str(item)
        else:
            out = str(raw)

    if isinstance(out, str):
        return out.strip()
    return str(out).strip()


def _cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _merge_from_files(
    llm: HuggingFacePipeline,
    tokenizer,
    chunk_run: ChunkRun,
    token_handler: TokenUsageHandler,
) -> str:
    """从磁盘读取 summary 文件，小批量 LLM 合并（可选模式）。"""
    merge_tpl = PromptTemplate.from_template(MERGE_USER_TEMPLATE)
    paths = chunk_run.list_summaries()
    level = 0

    while len(paths) > 1:
        level += 1
        print(f"  合并回合 {level}，待合并文件数: {len(paths)} ...")
        merged_paths: list[Path] = []
        merge_dir = chunk_run.run_dir / f"merge-L{level}"
        merge_dir.mkdir(exist_ok=True)

        for batch_idx, j in enumerate(range(0, len(paths), MERGE_BATCH), 1):
            batch_paths = paths[j : j + MERGE_BATCH]
            batch_texts = [
                p.read_text(encoding="utf-8").strip() for p in batch_paths
            ]
            mp = merge_tpl.format(partials="\n\n---\n\n".join(batch_texts))
            mp = _truncate_to_tokens(tokenizer, mp, MAX_INPUT_TOKENS)
            try:
                merged = invoke_llm(
                    llm, tokenizer, mp, token_handler, max_new_tokens=MAX_NEW_TOKENS_MERGE
                )
            except torch.OutOfMemoryError:
                print("  [OOM] 合并阶段显存不足，降级输入长度后重试 ...")
                _cuda_cleanup()
                mp = _truncate_to_tokens(
                    tokenizer, mp, int(MAX_INPUT_TOKENS * 0.7)
                )
                merged = invoke_llm(
                    llm, tokenizer, mp, token_handler, max_new_tokens=MAX_NEW_TOKENS_MERGE
                )

            out_path = merge_dir / f"merged-{batch_idx:03d}.txt"
            out_path.write_text(merged.strip() + "\n", encoding="utf-8")
            merged_paths.append(out_path)
            _cuda_cleanup()

        paths = merged_paths

    return paths[0].read_text(encoding="utf-8").strip()


def summarize_lecture(
    llm: HuggingFacePipeline,
    lecture: str,
    model_key: str,
    model_id: str,
    source_name: str,
) -> tuple[RunMetrics, int, Path]:
    reset_cuda_peak()
    mem_before = cuda_mem_mb()
    token_handler = TokenUsageHandler()
    chunk_run = ChunkRun.create(CHUNK_RUN_DIR, model_key, source_name)
    print(f"分段结果目录: {chunk_run.run_dir}")
    # 保存预处理后全文，便于核对模型输入
    (chunk_run.run_dir / "lecture-prepared.txt").write_text(lecture, encoding="utf-8")

    tokenizer = getattr(llm, "pipeline", None).tokenizer
    safe_in_tokens = max(512, int(MAX_INPUT_TOKENS) - 512)
    chunks = chunk_lecture_tokens(tokenizer, lecture, safe_in_tokens)

    start = time.perf_counter()
    with GpuSampler(interval=0.5) as sampler:
        print(f"讲义分 {len(chunks)} 段，逐段总结并落盘 ...")
        for i, chunk in enumerate(chunks, 1):
            print(f"  处理片段 {i}/{len(chunks)} ...")
            chunk_run.save_input(i, chunk)
            user_msg = USER_CHUNK_TEMPLATE.format(lecture=chunk)
            summary = invoke_llm(llm, tokenizer, user_msg, token_handler)
            saved = chunk_run.save_summary(i, summary)
            print(f"    已保存: {saved.name} ({len(summary):,} 字符)")
            _cuda_cleanup()

        if ASSEMBLE_MODE == "merge" and len(chunks) > 1:
            print("从磁盘读取各段，进行 LLM 合并 ...")
            answer = _merge_from_files(llm, tokenizer, chunk_run, token_handler)
        else:
            print("从磁盘拼接各段总结（不二次压缩）...")
            answer = chunk_run.assemble_concat()

        chunk_run.save_assembled(answer)

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
        question=(
            f"[讲义] {source_name} ({len(lecture):,} 字符, {len(chunks)} 段, "
            f"组装={ASSEMBLE_MODE})"
        ),
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
    return metrics, len(chunks), chunk_run.run_dir


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

    raw_len = len(lecture_path.read_text(encoding="utf-8"))
    print(f"\n已加载讲义: {lecture_path.name}")
    print(f"  原始字符数: {raw_len:,}")
    print(f"  预处理后: {len(lecture_text):,} 字符  {len(lecture_text.split(chr(10)+chr(10))):,} 段")
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
        metrics, num_chunks, chunk_run_dir = summarize_lecture(
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
        print(f"分段总结目录: {chunk_run_dir}")

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
                "assemble_mode": ASSEMBLE_MODE,
                "chunk_run_dir": str(chunk_run_dir),
            },
        )
        report_path = write_run_report(REPORT_DIR, report)
        print(f"运行报告已保存: {report_path}\n")


if __name__ == "__main__":
    main()
