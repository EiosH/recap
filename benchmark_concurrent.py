#!/usr/bin/env python3
"""
并发推理性能基准测试：对比 MODEL_REGISTRY 中 8b / 14b / 32b 档位在不同 token 规模与并发数下的表现。

测试维度：
  - 模型档位（默认 8b、14b、32b，见 config.MODEL_REGISTRY）
  - 输入 prompt token 数
  - 输出 max_new_tokens
  - 并发请求数

采集指标：
  - 单次/批量耗时、延迟分位数
  - 吞吐量（tokens/s）
  - GPU 显存峰值
  - 成功率与最大可承受并发数

跑完后在 reports/ 目录自动生成 Markdown + CSV 报告。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import (
    DO_SAMPLE,
    ENABLE_TF32,
    MAX_NEW_TOKENS,
    MODEL_REGISTRY,
    REPORT_DIR,
    TEMPERATURE,
    USE_FLASH_ATTENTION,
)
from gpu_monitor import (
    GpuSampler,
    collect_gpu_info_text,
    cuda_mem_mb,
    cuda_peak_mb,
    reset_cuda_peak,
)

try:
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
except ImportError:
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)

from simple_agent import _cuda_cleanup, _pipe_generate, format_chat_prompt

FILLER = (
    "Machine learning extracts patterns from data to support prediction and decision making. "
    "Neural networks learn hierarchical representations through gradient-based optimization. "
)

# 从小到大加载，减轻显存切换压力
_MODEL_ORDER = ("8b", "14b", "32b")
DEFAULT_MODELS = tuple(k for k in _MODEL_ORDER if k in MODEL_REGISTRY) or tuple(
    MODEL_REGISTRY.keys()
)
DEFAULT_INPUT_TOKENS = (256, 512, 1024, 2048)
DEFAULT_OUTPUT_TOKENS = (128, 256, 512)
DEFAULT_CONCURRENCY = (1, 2, 4, 8)


@dataclass
class RequestResult:
    request_id: int
    success: bool
    latency_sec: float
    output_chars: int
    error: str = ""


@dataclass
class BenchCase:
    model_key: str
    model_id: str
    quant: str
    role: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    actual_input_tokens: int
    success_count: int
    fail_count: int
    wall_time_sec: float
    latencies_sec: list[float] = field(default_factory=list)
    gpu_mem_before_mb: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    gpu_util_avg: float = 0.0
    gpu_util_max: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.fail_count == 0:
            return "ok"
        if self.success_count == 0:
            return "fail"
        return "partial"

    @property
    def avg_latency_sec(self) -> float:
        return statistics.mean(self.latencies_sec) if self.latencies_sec else 0.0

    @property
    def p50_latency_sec(self) -> float:
        return _percentile(self.latencies_sec, 50)

    @property
    def p95_latency_sec(self) -> float:
        return _percentile(self.latencies_sec, 95)

    @property
    def max_latency_sec(self) -> float:
        return max(self.latencies_sec) if self.latencies_sec else 0.0

    @property
    def throughput_tok_per_sec(self) -> float:
        if self.wall_time_sec <= 0:
            return 0.0
        total_out = self.success_count * self.output_tokens
        return total_out / self.wall_time_sec


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _prompt_token_count(tokenizer, user_content: str) -> int:
    prompt = format_chat_prompt(tokenizer, user_content)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def build_prompt_for_tokens(tokenizer, target_tokens: int) -> tuple[str, int]:
    """构造接近 target_tokens 的完整 chat prompt。"""
    user_prefix = "请用一句话概括以下内容的要点：\n\n"
    filler = ""
    prompt = ""
    tokens = 0

    for _ in range(10_000):
        filler += FILLER
        user_content = user_prefix + filler
        prompt = format_chat_prompt(tokenizer, user_content)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens >= target_tokens:
            break

    if tokens > target_tokens + 64:
        ids = tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    return prompt, tokens


def _resolve_dtype(name: str):
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _setup_cuda() -> None:
    if not torch.cuda.is_available():
        return
    if ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


def build_benchmark_llm(model_key: str) -> tuple[HuggingFacePipeline, float, str]:
    """按 MODEL_REGISTRY 加载模型（支持 AWQ / bitsandbytes INT4）。"""
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
        model_kwargs["torch_dtype"] = dtype
        quant_desc = "AWQ INT4"
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
    hf_llm.pipeline.tokenizer = tokenizer
    return hf_llm, load_sec, quant_desc


def _run_one(
    llm,
    prompt: str,
    max_new_tokens: int,
    request_id: int,
    lock: threading.Lock | None,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        if lock is not None:
            with lock:
                text = _pipe_generate(llm, prompt, max_new_tokens=max_new_tokens)
        else:
            text = _pipe_generate(llm, prompt, max_new_tokens=max_new_tokens)
        latency = time.perf_counter() - t0
        return RequestResult(
            request_id=request_id,
            success=True,
            latency_sec=latency,
            output_chars=len(text),
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            success=False,
            latency_sec=time.perf_counter() - t0,
            output_chars=0,
            error=str(e),
        )


def run_bench_case(
    llm,
    tokenizer,
    model_key: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    serialize: bool,
) -> BenchCase:
    cfg = MODEL_REGISTRY[model_key]
    prompt, actual_tokens = build_prompt_for_tokens(tokenizer, input_tokens)

    lock = threading.Lock() if serialize else None
    mem_before = cuda_mem_mb()
    reset_cuda_peak()

    results: list[RequestResult] = []
    t_wall = time.perf_counter()

    with GpuSampler(interval=0.3) as sampler:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _run_one, llm, prompt, output_tokens, i, lock
                )
                for i in range(concurrency)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

    wall = time.perf_counter() - t_wall
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ok = [r for r in results if r.success]
    bad = [r for r in results if not r.success]

    return BenchCase(
        model_key=model_key,
        model_id=cfg["model_id"],
        quant=cfg.get("quant", ""),
        role=cfg.get("role", ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        actual_input_tokens=actual_tokens,
        success_count=len(ok),
        fail_count=len(bad),
        wall_time_sec=wall,
        latencies_sec=[r.latency_sec for r in ok],
        gpu_mem_before_mb=mem_before,
        gpu_mem_peak_mb=cuda_peak_mb(),
        gpu_util_avg=(
            sum(sampler.samples) / len(sampler.samples) if sampler.samples else 0.0
        ),
        gpu_util_max=max(sampler.samples) if sampler.samples else 0.0,
        errors=[r.error for r in bad if r.error],
    )


def find_max_concurrency(
    llm,
    tokenizer,
    model_key: str,
    input_tokens: int,
    output_tokens: int,
    serialize: bool,
    start: int = 1,
    max_try: int = 32,
) -> tuple[int, BenchCase | None]:
    """递增并发直到全部失败，返回上次全部成功的并发数及对应结果。"""
    last_ok: BenchCase | None = None
    last_ok_n = 0
    n = start

    while n <= max_try:
        case = run_bench_case(
            llm, tokenizer, model_key, input_tokens, output_tokens, n, serialize
        )
        if case.fail_count == 0:
            last_ok = case
            last_ok_n = n
            n *= 2
            continue
        break

    return last_ok_n, last_ok


def unload_llm(llm) -> None:
    del llm
    _cuda_cleanup()


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [_row(headers), sep]
    lines.extend(_row(r) for r in rows)
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    cases: list[BenchCase],
    max_conc_rows: list[dict],
    meta: dict,
    stamp: str,
) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    base = f"benchmark-concurrent-{stamp}"

    md_path = report_dir / f"{base}.md"
    csv_path = report_dir / f"{base}.csv"
    json_path = report_dir / f"{base}.json"

    # --- CSV ---
    fieldnames = [
        "model_key",
        "model_id",
        "quant",
        "role",
        "input_tokens",
        "actual_input_tokens",
        "output_tokens",
        "concurrency",
        "status",
        "success_count",
        "fail_count",
        "wall_time_sec",
        "avg_latency_sec",
        "p50_latency_sec",
        "p95_latency_sec",
        "max_latency_sec",
        "throughput_tok_per_sec",
        "gpu_mem_before_mb",
        "gpu_mem_peak_mb",
        "gpu_util_avg",
        "gpu_util_max",
        "errors",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "model_key": c.model_key,
                    "model_id": c.model_id,
                    "quant": c.quant,
                    "role": c.role,
                    "input_tokens": c.input_tokens,
                    "actual_input_tokens": c.actual_input_tokens,
                    "output_tokens": c.output_tokens,
                    "concurrency": c.concurrency,
                    "status": c.status,
                    "success_count": c.success_count,
                    "fail_count": c.fail_count,
                    "wall_time_sec": f"{c.wall_time_sec:.3f}",
                    "avg_latency_sec": f"{c.avg_latency_sec:.3f}",
                    "p50_latency_sec": f"{c.p50_latency_sec:.3f}",
                    "p95_latency_sec": f"{c.p95_latency_sec:.3f}",
                    "max_latency_sec": f"{c.max_latency_sec:.3f}",
                    "throughput_tok_per_sec": f"{c.throughput_tok_per_sec:.1f}",
                    "gpu_mem_before_mb": f"{c.gpu_mem_before_mb:.0f}",
                    "gpu_mem_peak_mb": f"{c.gpu_mem_peak_mb:.0f}",
                    "gpu_util_avg": f"{c.gpu_util_avg:.1f}",
                    "gpu_util_max": f"{c.gpu_util_max:.1f}",
                    "errors": "; ".join(c.errors[:3]),
                }
            )

    # --- JSON ---
    payload = {
        "meta": meta,
        "cases": [asdict(c) for c in cases],
        "max_concurrency": max_conc_rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Markdown ---
    detail_rows = [
        [
            c.model_key,
            str(c.input_tokens),
            str(c.output_tokens),
            str(c.concurrency),
            c.status,
            f"{c.wall_time_sec:.2f}s",
            f"{c.avg_latency_sec:.2f}s",
            f"{c.p95_latency_sec:.2f}s",
            f"{c.throughput_tok_per_sec:.0f}",
            f"{c.gpu_mem_peak_mb:.0f}",
            f"{c.success_count}/{c.concurrency}",
        ]
        for c in cases
    ]

    max_rows = [
        [
            r["model_key"],
            str(r["input_tokens"]),
            str(r["output_tokens"]),
            str(r["max_concurrency"]),
            f"{r.get('wall_time_sec', 0):.2f}s",
            f"{r.get('avg_latency_sec', 0):.2f}s",
            f"{r.get('gpu_mem_peak_mb', 0):.0f}",
        ]
        for r in max_conc_rows
    ]

    lines = [
        "# 并发推理性能基准报告",
        "",
        f"- **生成时间**: {meta['finished_at']}",
        f"- **测试模型**: {', '.join(meta['models'])}",
        f"- **输入 token 档位**: {meta['input_tokens']}",
        f"- **输出 token 档位**: {meta['output_tokens']}",
        f"- **并发档位**: {meta['concurrency']}",
        f"- **推理串行化 (单实例队列)**: {meta['serialize']}",
        f"- **探测最大并发**: {meta['find_max']}",
        "",
        "## 模型配置 (MODEL_REGISTRY)",
        "",
        _fmt_table(
            ["档位", "HuggingFace ID", "量化", "角色"],
            [
                [
                    k,
                    v["model_id"],
                    v.get("quant", "-"),
                    v.get("role", "-"),
                ]
                for k, v in MODEL_REGISTRY.items()
                if k in meta["models"]
            ],
        ),
        "",
        "## 显卡快照",
        "",
        "```",
        meta.get("gpu_snapshot", "").strip() or "(无)",
        "```",
        "",
        "## 全量测试结果",
        "",
        _fmt_table(
            [
                "模型",
                "输入tok",
                "输出tok",
                "并发",
                "状态",
                "总耗时",
                "均延迟",
                "P95延迟",
                "吞吐tok/s",
                "显存峰值MB",
                "成功数",
            ],
            detail_rows,
        ),
        "",
    ]

    if max_rows:
        lines.extend(
            [
                "## 最大可承受并发（全部成功）",
                "",
                _fmt_table(
                    [
                        "模型",
                        "输入tok",
                        "输出tok",
                        "最大并发",
                        "总耗时",
                        "均延迟",
                        "显存峰值MB",
                    ],
                    max_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 文件",
            "",
            f"- Markdown: `{md_path}`",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
            "",
            "---",
            "*由 benchmark_concurrent.py 自动生成*",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path, json_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="测试 MODEL_REGISTRY 中各档位在不同 token 与并发下的推理性能，并生成报告。"
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(MODEL_REGISTRY.keys()),
        help=f"要测试的模型档位（默认: {' '.join(DEFAULT_MODELS)}）",
    )
    p.add_argument(
        "--input-tokens",
        nargs="+",
        type=int,
        default=list(DEFAULT_INPUT_TOKENS),
        help="目标输入 prompt token 数列表",
    )
    p.add_argument(
        "--output-tokens",
        nargs="+",
        type=int,
        default=list(DEFAULT_OUTPUT_TOKENS),
        help="每次请求的 max_new_tokens 列表",
    )
    p.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONCURRENCY),
        help="并发请求数列表",
    )
    p.add_argument(
        "--find-max",
        action="store_true",
        help="对每个 (模型, 输入tok, 输出tok) 递增探测最大可承受并发",
    )
    p.add_argument(
        "--no-serialize",
        action="store_true",
        help="不串行化推理（多线程同时调 pipeline，可能不稳定，仅用于压测）",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：更少档位组合 (256/1024 输入, 128/256 输出, 并发 1/4)",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="报告输出目录",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    serialize = not args.no_serialize

    models: list[str] = args.models
    input_tokens: list[int] = args.input_tokens
    output_tokens: list[int] = args.output_tokens
    concurrency_levels: list[int] = args.concurrency

    if args.quick:
        input_tokens = [256, 1024]
        output_tokens = [128, 256]
        concurrency_levels = [1, 4]

    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    gpu_snapshot = collect_gpu_info_text()

    total_cases = len(models) * len(input_tokens) * len(output_tokens)
    if not args.find_max:
        total_cases *= len(concurrency_levels)

    print("=" * 60)
    print("并发推理性能基准测试")
    print("=" * 60)
    for k in models:
        cfg = MODEL_REGISTRY[k]
        print(f"  [{k}] {cfg['model_id']}  ({cfg.get('quant', 'fp')}, {cfg.get('role', '-')})")
    print(f"输入 token: {input_tokens}")
    print(f"输出 token: {output_tokens}")
    print(f"并发: {concurrency_levels if not args.find_max else '(递增探测)'}")
    print(f"串行队列模式: {serialize}")
    print(f"预计用例数: {total_cases}")
    print("=" * 60)

    all_cases: list[BenchCase] = []
    max_conc_rows: list[dict] = []
    case_idx = 0

    for model_key in models:
        print(f"\n>>> 加载模型 [{model_key}] ...")
        llm, load_sec, quant_desc = build_benchmark_llm(model_key)
        tokenizer = llm.pipeline.tokenizer
        print(f"    量化: {quant_desc}，加载 {load_sec:.1f}s")

        for in_tok in input_tokens:
            for out_tok in output_tokens:
                if args.find_max:
                    case_idx += 1
                    print(
                        f"\n[{case_idx}] {model_key} | in={in_tok} out={out_tok} | 探测最大并发 ..."
                    )
                    max_n, case = find_max_concurrency(
                        llm,
                        tokenizer,
                        model_key,
                        in_tok,
                        out_tok,
                        serialize,
                    )
                    if case is not None:
                        all_cases.append(case)
                    max_conc_rows.append(
                        {
                            "model_key": model_key,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "max_concurrency": max_n,
                            "wall_time_sec": case.wall_time_sec if case else 0,
                            "avg_latency_sec": case.avg_latency_sec if case else 0,
                            "gpu_mem_peak_mb": case.gpu_mem_peak_mb if case else 0,
                        }
                    )
                    print(f"    最大并发(全成功): {max_n}")
                    continue

                for conc in concurrency_levels:
                    case_idx += 1
                    print(
                        f"\n[{case_idx}/{total_cases}] {model_key} | "
                        f"in={in_tok} out={out_tok} conc={conc} ..."
                    )
                    case = run_bench_case(
                        llm,
                        tokenizer,
                        model_key,
                        in_tok,
                        out_tok,
                        conc,
                        serialize,
                    )
                    all_cases.append(case)
                    print(
                        f"    {case.status} | wall={case.wall_time_sec:.2f}s "
                        f"avg={case.avg_latency_sec:.2f}s p95={case.p95_latency_sec:.2f}s "
                        f"tok/s={case.throughput_tok_per_sec:.0f} "
                        f"peak={case.gpu_mem_peak_mb:.0f}MB "
                        f"ok={case.success_count}/{conc}"
                    )
                    if case.errors:
                        print(f"    错误: {case.errors[0][:120]}")

        print(f"\n>>> 卸载模型 [{model_key}] ...")
        unload_llm(llm)
        gc.collect()

    finished = datetime.now()
    meta = {
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": (finished - started).total_seconds(),
        "models": models,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency_levels,
        "serialize": serialize,
        "find_max": args.find_max,
        "gpu_snapshot": gpu_snapshot,
    }

    md_path, csv_path, json_path = write_reports(
        args.report_dir, all_cases, max_conc_rows, meta, stamp
    )

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"总耗时: {meta['elapsed_sec']:.1f}s")
    print(f"Markdown 报告: {md_path}")
    print(f"CSV 报告:     {csv_path}")
    print(f"JSON 原始数据: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
