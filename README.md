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

## 模型档位（可在 `config.py` 修改）

| 档位 | 默认 HuggingFace ID | 参数量级 |
|------|---------------------|----------|
| 30b  | Qwen/Qwen2.5-32B-Instruct | ~32B |
| 20b  | google/gemma-2-27b-it     | ~27B |
| 10b  | Qwen/Qwen2.5-14B-Instruct | ~14B |

默认开启 **4bit 量化**（`LOAD_IN_4BIT = True`），降低显存占用。显存足够可在 `config.py` 关闭。

## 运行

```bash
python simple_agent.py
```

启动后自动加载 `data/subtitles-519105.txt`（可在 `config.py` 修改 `LECTURE_TXT_NAME`）。若只有 `.vtt` 会自动先提取。

### 交互说明

1. 输入 `30b` / `20b` / `10b` 选择模型档位。
2. 输入 `run` 生成课程知识提纲（讲义过长时会自动分块再合并）。
3. `gpu` — 刷新显卡信息；`quit` — 退出。

提纲保存为 `data/subtitles-519105-outline.txt`（与讲义同名加 `-outline` 后缀）。

每次运行会输出耗时、显存、GPU 利用率、Token 等指标。

## 注意

- 三档模型**不会同时加载**；切换档位会释放显存后重新加载。
- 32B/27B 即使 4bit 仍需要较大显存；显存不足请改用小模型 ID 或换 `10b` 档位。
- 首次运行会从 Hugging Face 下载权重，耗时较长。
