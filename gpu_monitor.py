"""GPU 信息与运行指标采集。"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class RunMetrics:
    model_key: str
    model_id: str
    question: str
    answer: str
    elapsed_sec: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    gpu_mem_before_mb: float = 0.0
    gpu_mem_after_mb: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    gpu_util_samples: list[float] = field(default_factory=list)

    @property
    def gpu_util_avg(self) -> float:
        if not self.gpu_util_samples:
            return 0.0
        return sum(self.gpu_util_samples) / len(self.gpu_util_samples)

    @property
    def gpu_util_max(self) -> float:
        return max(self.gpu_util_samples) if self.gpu_util_samples else 0.0


def _nvml_init() -> bool:
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def collect_gpu_info_text() -> str:
    """返回显卡信息文本（用于 report）。"""
    import io

    buf = io.StringIO()

    def _line(s: str = "") -> None:
        buf.write(s + "\n")

    _line("=" * 60)
    _line("当前显卡 / CUDA 信息")
    _line("=" * 60)

    if torch is not None:
        _line(f"PyTorch 版本: {torch.__version__}")
        _line(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            _line(f"CUDA 版本: {torch.version.cuda}")
            _line(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                _line(f"  [{i}] {props.name}")
                _line(f"      显存总量: {total_gb:.2f} GB")
                _line(f"      计算能力: {props.major}.{props.minor}")
    else:
        _line("未安装 PyTorch")

    if _nvml_init() and pynvml is not None:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                _line(f"\nNVML [{i}] {name}")
                _line(
                    f"  已用显存: {mem.used / 1024**2:.0f} MB / "
                    f"{mem.total / 1024**2:.0f} MB"
                )
                _line(f"  GPU 利用率: {util.gpu}%  显存利用率: {util.memory}%")
                _line(f"  温度: {temp}°C")
        except Exception as e:
            _line(f"NVML 读取失败: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    else:
        smi = _nvidia_smi_text()
        if smi:
            _line(smi)

    _line("=" * 60)
    return buf.getvalue()


def _nvidia_smi_text() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        )
        lines = ["nvidia-smi:"]
        lines.extend(f"  {line}" for line in out.strip().splitlines())
        return "\n".join(lines)
    except (FileNotFoundError, subprocess.SubprocessError):
        return "无法读取 GPU（无 NVML / nvidia-smi）"


def print_gpu_info() -> None:
    print("\n" + collect_gpu_info_text())


def cuda_mem_mb(device: int = 0) -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**2)


def cuda_peak_mb(device: int = 0) -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_cuda_peak(device: int = 0) -> None:
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def sample_gpu_util(device_index: int = 0) -> float:
    if not _nvml_init() or pynvml is None:
        return 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class GpuSampler:
    """在推理期间后台采样 GPU 利用率。"""

    def __init__(self, interval: float = 0.5, device_index: int = 0):
        self.interval = interval
        self.device_index = device_index
        self.samples: list[float] = []
        self._running = False

    def __enter__(self) -> "GpuSampler":
        self._running = True
        self.samples = []
        import threading

        def _loop():
            while self._running:
                self.samples.append(sample_gpu_util(self.device_index))
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)


def print_run_metrics(m: RunMetrics) -> None:
    print("\n" + "-" * 60)
    print("本次运行指标")
    print("-" * 60)
    print(f"模型档位: {m.model_key}  ({m.model_id})")
    print(f"耗时: {m.elapsed_sec:.2f} 秒")
    print(f"显存 (MB) — 运行前: {m.gpu_mem_before_mb:.0f}  运行后: {m.gpu_mem_after_mb:.0f}  峰值: {m.gpu_mem_peak_mb:.0f}")
    print(f"GPU 利用率 — 平均: {m.gpu_util_avg:.1f}%  峰值: {m.gpu_util_max:.1f}%")
    print(f"Token — 输入: {m.prompt_tokens}  输出: {m.completion_tokens}  合计: {m.total_tokens}")
    print("-" * 60)
    print("回答:\n")
    print(m.answer)
    print("-" * 60 + "\n")
