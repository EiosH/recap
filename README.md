# LangChain + Hugging Face 课程提纲生成

从 `data/` 目录读取讲义 txt（由 WebVTT 字幕提取），用本地大模型生成课程知识提纲，并打印耗时、显存、GPU 利用率、Token 等指标。

## 字幕转纯文本

```bash
python extract_subtitles.py
```

识别 `data/*.vtt`（WebVTT），去掉时间轴与 cue 编号，输出同名 `.txt`。已处理示例：`subtitles-519105.vtt` → `subtitles-519105.txt`。

## 环境

- Python 3.10+
- NVIDIA GPU + CUDA（推荐）
- 已安装 `nvidia-smi`（可选，用于备用 GPU 信息）

```bash
cd /Users/wangyouhe/Documents/cdm/recap
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

访问 gated 模型时：

```bash
export HF_TOKEN=你的_huggingface_token
```

## 模型档位（RTX 4090 24GB 优化，见 `config.py`）

| 档位 | 模型 | 4090 策略 |
|------|------|-----------|
| 30b  | Qwen2.5-32B-Instruct | 4bit NF4（约 18–20GB 显存） |
| 20b  | Qwen2.5-14B-Instruct | 4bit NF4（长讲义稳妥） |
| 10b  | Qwen2.5-7B-Instruct  | BF16 全精度（最快） |

另启用 **TF32**、**Flash Attention 2**（已列入 `requirements.txt`）。

## 运行

```bash
python simple_agent.py
```

启动后自动加载 `data/subtitles-519105.txt`（可在 `config.py` 修改 `LECTURE_TXT_NAME`）。若只有 `.vtt` 会自动先提取。

### 交互说明

1. 输入 `30b` / `20b` / `10b` 选择模型档位。
2. 输入 `run` 生成课程知识提纲（讲义过长时会自动分块再合并）。
3. `gpu` — 刷新显卡信息；`quit` — 退出。

- 提纲：`data/subtitles-519105-outline.txt`
- 完整报告：`reports/run-YYYYMMDD-HHMMSS-{档位}.md`（含耗时、显存、GPU、Token、配置、提纲全文）

每次运行会同时在终端打印指标，并写入上述 report。

## 注意

- 三档模型**不会同时加载**；切换档位会释放显存后重新加载。
- 32B/27B 即使 4bit 仍需要较大显存；显存不足请改用小模型 ID 或换 `10b` 档位。
- 首次运行会从 Hugging Face 下载权重，耗时较长。
