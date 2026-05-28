# VLA Model (Sim2Real)

基于 GraspVLA 架构的 Vision-Language-Action (VLA) 模型，用于机器人动作生成与 sim2real 迁移。

## 架构

1. **ViT (Vision Transformer)**：处理图像输入，提取视觉特征  
2. **VLM (预训练大模型)**：融合 ViT 特征与文本 token，进行推理  
3. **Action Expert**：基于 Flow Matching 从 VLM 输出生成 action chunk  

```
Images → ViT → Projector → VLM (+ text tokens) → Action Expert (Flow Matching) → Action Chunk
```

## 项目结构

```
sim2real/
├── vla_model/
│   ├── config/          # 配置
│   ├── type/            # 数据类型
│   ├── model/
│   │   ├── backbone_2d/  # ViT 视觉编码器
│   │   ├── backbone_llm/ # VLM 语言模型
│   │   └── vla/         # 主模型、Projector、Flow Matching、Action Expert
│   ├── data/            # 数据预处理、Tokenizer、Collator
│   └── utils/
├── scripts/
│   ├── train.py         # 训练脚本
│   └── inference.py     # 推理脚本
├── requirements.txt
└── README.md
```

## 安装

```bash
cd sim2real
pip install -r requirements.txt
```

## 启动模型

### 1. 安装依赖

```bash
cd sim2real
pip install -r requirements.txt
```

### 2. 下载 SmolLM2-360M

模型首次运行时会自动从 HuggingFace 下载，也可手动预下载：

```bash
# 使用 huggingface-cli（需先 pip install huggingface_hub）
huggingface-cli download HuggingFaceTB/SmolLM2-360M --local-dir ./models/SmolLM2-360M
```

若在国内，可设置镜像加速：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 启动推理

**方式一：Python 脚本**
```bash
cd sim2real
python -m scripts.inference --checkpoint path/to/checkpoint.pt --device cuda:0
```

**方式二：在代码中调用**
```python
from vla_model import VLAAgent

agent = VLAAgent(path="path/to/checkpoint.pt", device="cuda:0")
# 无 checkpoint 时也可加载默认配置进行测试
# agent = VLAAgent(path=None, device="cuda:0")
```

### 4. 显存需求

SmolLM2-360M 约需 720MB（bf16），整体 VLA 模型（含 ViT、Projector、Action Expert）约需 2–4GB 显存。

---

## 使用

### 推理

```python
from vla_model import VLAAgent
from vla_model.type import RawVLAData
import numpy as np

agent = VLAAgent(path="path/to/checkpoint.pt", device="cuda:0")

raw = RawVLAData(
    instruction="pick up the red block",
    images={
        "front": np.array(...),  # (H, W, 3) uint8
        "side": np.array(...),
    },
    proprio=np.array(...),  # (proprio_len, proprio_dim)
)

result = agent.sample_action(raw)
action = result["action"]  # (action_len, action_dim)
```

### 命令行推理

```bash
python -m scripts.inference --checkpoint path/to/ckpt.pt --instruction "grasp the object"
```

## 配置

主要配置在 `vla_model/config/define.py`：

- `ViTConfig`: ViT 模型名称、图像尺寸、预训练  
- `LLMConfig`: VLM 名称（如 Qwen2-0.5B）、最大长度  
- `FlowMatchingConfig`: Flow Matching 参数  
- `ActionExpertConfig`: Action Expert 缩放系数  

## 依赖

- PyTorch 2.0+
- timm (Vision Transformer)
- transformers (HuggingFace)
- 可选：safetensors

## 参考

基于 [GraspVLA](https://github.com/.../GraspVLA) 的实现逻辑构建。
