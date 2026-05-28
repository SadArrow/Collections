# myVLA inference_viz

这个目录用于存放一次次推理验证时导出的可视化结果（输入图像/视频、自然语言输入、long-term memory 迭代、actions 等）。

运行示例（MEM）：

```bash
python myVLA/run_pi05_mem_inference.py --checkpoint_dir myVLA/pi05_droid_pytorch --num_steps 1 --video_window 4 --mem_steps 1 --hl_vlm_dir myVLA/pretrained_vlm/google_paligemma-3b-mix-224-bfloat16 --hl_device cpu --viz
```

导出后会在 `myVLA/inference_viz/<run_name>/` 下生成：
- `meta.json`：运行参数与环境信息
- `report.md`：可视化总览（包含图片/动图）
- `step_000/`：每一步的输入/记忆/动作导出

