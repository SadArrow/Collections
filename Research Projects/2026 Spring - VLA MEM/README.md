# myVLA: pi0.5 (pi05_droid) PyTorch 推理（独立运行）

这个目录已经包含 **pi0.5（`pi05_droid`）的 PyTorch 推理代码 + 所需资源**，运行时**不依赖** `openpi-main/openpi-main`。

## 目录结构（关键文件）

- `myVLA/pi05_droid_pytorch/model.safetensors`：模型权重（约 7.2GB）
- `myVLA/pi05_droid_pytorch/config.json`：模型结构参数
- `myVLA/pi05_droid_pytorch/assets/droid/norm_stats.json`：归一化统计（推理必需）
- `myVLA/assets/paligemma_tokenizer.model`：PaliGemma SentencePiece tokenizer（推理必需）
- `myVLA/vendor/transformers_replace/`：Transformers 补丁（pi0.5 PyTorch 需要）
- `myVLA/myvla_pi05/`：pi0.5 PyTorch 推理实现（已从 openpi 代码中抽取并去掉 JAX 依赖）
- `myVLA/run_pi05_inference.py`：推理入口脚本

## 运行步骤（WSL/Linux 推荐）

### 0) 准备 Python 环境

建议使用 **Python 3.10+**，并在虚拟环境中安装依赖。

最小依赖：
- `torch`（建议 GPU 版本）
- `transformers==4.53.2`
- `safetensors`
- `sentencepiece`
- `numpy`
- `einops`

你可以用 `pip` 安装（示例）：

```bash
pip install --upgrade pip
pip install "transformers==4.53.2" safetensors sentencepiece numpy einops
# torch 请按你的 CUDA/CPU 情况安装（参考 PyTorch 官方安装命令）
```

### 1)（一次性）应用 transformers_replace 补丁

pi0.5 的 PyTorch 实现在 Transformers 上需要一个小补丁（`transformers_replace`）。

本项目会在运行时自动检测，如果缺失会尝试把：

`myVLA/vendor/transformers_replace/*`

复制到你当前环境的：

`.../site-packages/transformers/`

如果自动复制失败（通常是权限问题），请手动执行（在你的 venv 里）：

```bash
python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).resolve().parent)"
# 假设输出是 <TRANSFORMERS_DIR>
cp -r myVLA/vendor/transformers_replace/* <TRANSFORMERS_DIR>/
```

### 2) 运行推理（使用本地 checkpoint）

在 WSL/Linux 中：

```bash
cd myVLA
python3 run_pi05_inference.py
```

或用脚本：

```bash
cd myVLA
./run_test_wsl.sh
```

常用参数：

```bash
python3 run_pi05_inference.py --checkpoint_dir ./pi05_droid_pytorch --device cuda --num_steps 10
```

### 3) 首次运行很慢？（torch.compile）

默认 `PI0Pytorch` 会启用 `torch.compile`，第一次会有较长编译/调优时间。

如果你只想先验证能跑通，可临时关闭：

```bash
export MYVLA_DISABLE_TORCH_COMPILE=1
python3 run_pi05_inference.py
```

## Windows 运行（推荐走 WSL）

在 Windows PowerShell 中（会调用 WSL）：

```powershell
cd myVLA
.\run_test.ps1
```

如果你需要指定 WSL 发行版：

```powershell
$env:MYVLA_WSL_DISTRO = "NVIDIA-SDKM-Ubuntu-24.04"
.\run_test.ps1
```

## 推理流程解释（做了什么）

`run_pi05_inference.py` 最终会调用 `myvla_pi05.Pi05DroidPolicy`，核心流程：

1. 读取 checkpoint：`model.safetensors` + `config.json`
2. 读取归一化统计：`assets/droid/norm_stats.json`
3. 读取 tokenizer：`assets/paligemma_tokenizer.model`
4. 把一个 DROID 格式的示例（随机图片 + state + prompt）转换成模型输入：
   - state padding 到 32 维
   - state 用 `norm_stats` 做 z-score normalize
   - 将 `prompt + normalized_state` 用 SentencePiece tokenization（pi0.5 的离散 state 输入格式）
5. 调用 `PI0Pytorch.sample_actions(...)` 得到 `(action_horizon, 32)` 的动作
6. 用 `norm_stats` 对动作做 unnormalize，并返回前 8 维（DROID 输出约定）

## 旧的 openpi 脚本

`myVLA/legacy_openpi/` 里保留了早期依赖 `openpi-main/openpi-main` 的下载/转换脚本，仅供参考；**不影响**本目录独立推理。

## MEM（Multi-Scale Embodied Memory）修改版：长/短期记忆

基于论文 `MEM/Mem.pdf`，在现有 pi0.5 的基础上加入两个 memory processor：

1) **Long-term memory processor（语言长期记忆 + subtask）**
- 位置：`myVLA/myvla_mem/policy.py`
- 实现：`myVLA/myvla_mem/long_term.py`（`PretrainedVlmLongTermMemoryProcessor`）
- 行为：维护 `language_memory (m_t)`，每步用“预训练 VLM（SigLIP+Gemma，例如 PaliGemma）”根据 `o_t + m_t + g` 生成：
  - 更新后的 `m_{t+1}`
  - 下一步 `subtask l_{t+1}`
  并把 `g + l_{t+1}` 作为 prompt 输入到低层 pi0.5 policy。
- 注意：这里要求 **单独的预训练 VLM 权重**（不能用已 fine-tune 的 pi0.5 权重）。需要你提供本地 `--hl_vlm_dir` 路径/模型名。

2) **Short-term memory processor（Video memory encoder）**
- 位置：SigLIP 视觉塔内部（低层 pi0.5 的 ViT 视觉编码器）
- 实现：`myVLA/vendor/transformers_replace/models/siglip/modeling_siglip.py`（`SiglipVisionTransformer._forward_video`）
- 行为：当输入 `pixel_values` 为 `[B, T, C, H, W]` 时：
  - 对每帧独立 patchify
  - 每 4 层交错加入 **causal temporal attention**（同一 patch 跨时间）+ **bidirectional spatial attention**（同一帧跨空间）
  - 在上层丢弃过去帧 token，仅输出当前帧 token 给 VLM backbone
- 触发方式：把观测图片堆叠成视频窗口（`--video_window > 1`），并作为图片输入给模型。

### 运行（短期记忆 / Video encoder）

```bash
cd myVLA
python3 run_pi05_inference.py --video_window 8
```

### 运行（长+短期记忆 / MEM demo）

```bash
cd myVLA
python3 run_pi05_mem_inference.py --video_window 8 --mem_steps 2 --hl_vlm_dir <PRETRAINED_VLM_DIR>
```

其中 `<PRETRAINED_VLM_DIR>` 是一个 SigLIP+Gemma 的预训练 VLM（如 PaliGemma）在本地的目录（或 transformers 支持的模型 id）。
如果显存不足，可以把高层 VLM 放到 CPU：加 `--hl_device cpu`。

### 下载预训练 VLM（PaliGemma，SigLIP+Gemma）

MEM 的 long-term memory processor 需要一个**独立的预训练 VLM 权重**（不使用 pi0.5 的 fine-tune 权重）。

我们提供了下载脚本（会下载到 `myVLA/pretrained_vlm/`）：

```bash
python3 myVLA/scripts/download_pretrained_vlm.py --repo_id google/paligemma-3b-mix-224 --revision bfloat16
```

注意：PaliGemma/Gemma 通常是 gated 模型，需要你在 Hugging Face 上接受 license，并提供 `HF_TOKEN`（或脚本参数 `--token`）。

