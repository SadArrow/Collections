# Isaac Sim (standalone) + myVLA (pi0.5 + memory)

本目录提供一个 **headless**（无 GUI）示例脚本：在 Isaac Sim 中构建 **双机械臂 + particle cloth（“shirt”）** 场景，并把仿真相机观测喂给 `myVLA` 的 **pi0.5 + memory** 结构，得到 `action chunk` 后回写到两只机械臂（关节目标）中。

## 推荐运行方式（Windows, headless）

Windows 上 Isaac Sim（Kit）进程内导入 `sentencepiece` 可能会硬崩溃，因此推荐使用 **两进程 RPC 架构**：
- Isaac Sim 只负责仿真、相机采样、执行动作
- `myVLA`（系统 Python）负责加载 pi0.5(+memory) 并推理

### 0) 先做一次 RPC smoke test（不需要 Isaac Sim）

```powershell
$env:MYVLA_DISABLE_TORCH_COMPILE=1
python myVLA/isaac_sim/rpc_smoketest.py --spawn_server --device cuda:0
```

成功时会打印 `viz_run_dir`，并在 `myVLA/isaac_sim_viz/<run>/` 看到 `step_000/*` 与 `final_actions.npy` 等输出。

### 1) 启动推理 RPC server（系统 Python 终端 A）

```powershell
$env:MYVLA_DISABLE_TORCH_COMPILE=1
python myVLA/isaac_sim/policy_rpc_server.py --port 5555 --device cuda:0 --checkpoint_dir myVLA/pi05_droid_pytorch
```

启用 long-term memory（需要预训练 VLM 权重目录）：

```powershell
$env:MYVLA_DISABLE_TORCH_COMPILE=1
python myVLA/isaac_sim/policy_rpc_server.py --port 5555 --device cuda:0 --checkpoint_dir myVLA/pi05_droid_pytorch `
  --hl_vlm_dir myVLA/pretrained_vlm/google_paligemma-3b-mix-224-bfloat16 --hl_device cpu
```

### 2) 启动 Isaac Sim headless（Isaac Sim 终端 B）

建议用 Isaac Sim 自带 Python 运行：

```powershell
cd issac-sim
.\python.bat ..\myVLA\isaac_sim\fold_shirt_dual_arm_mem_pi05.py --headless --policy_mode rpc --rpc_port 5555 --mem_steps 1 --video_window 4 --no_cloth
```

如果你确认 cloth 能稳定运行，再去掉 `--no_cloth`：

```powershell
cd issac-sim
.\python.bat ..\myVLA\isaac_sim\fold_shirt_dual_arm_mem_pi05.py --headless --policy_mode rpc --rpc_port 5555 --mem_steps 1 --video_window 4
```

> 如果遇到 `rpc timeout`，把 Isaac 侧 `--rpc_timeout_s`（以及 server 侧 `--timeout_s`）设大一些。

脚本会把输入/记忆/动作导出到：
- `myVLA/isaac_sim_viz/<run_name>/`

> 注：当前策略并未“真正学会”在 Isaac Sim 中叠衣服；该脚本的目标是先把 **观测 -> VLA -> action -> 仿真** 的链路跑通，并能把中间结果可视化落盘。
