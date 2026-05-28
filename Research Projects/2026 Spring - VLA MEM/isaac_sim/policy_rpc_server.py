from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import importlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any


def _sanitize_runtime_pydeps_imports() -> None:
    markers = (
        "WorldModelDiffusionVlaRuntime/pydeps",
        "WorldModelDiffusionVlaRuntime\\pydeps",
    )
    had_runtime_overlay = any(any(marker in os.fspath(item) for marker in markers) for item in sys.path)

    kept_sys_path: list[str] = []
    for item in sys.path:
        text = os.fspath(item)
        if any(marker in text for marker in markers):
            continue
        kept_sys_path.append(item)
    sys.path[:] = kept_sys_path

    raw_pythonpath = str(os.environ.get("PYTHONPATH", "") or "")
    if raw_pythonpath:
        kept = [
            token
            for token in raw_pythonpath.split(os.pathsep)
            if token and not any(marker in token for marker in markers)
        ]
        if len(kept) != len([token for token in raw_pythonpath.split(os.pathsep) if token]):
            had_runtime_overlay = True
        if kept:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            os.environ.pop("PYTHONPATH", None)

    if had_runtime_overlay:
        for name in tuple(sys.modules):
            if (
                name == "numpy"
                or name.startswith("numpy.")
                or name == "scipy"
                or name.startswith("scipy.")
                or name == "sklearn"
                or name.startswith("sklearn.")
                or name == "transformers"
                or name.startswith("transformers.")
                or name == "huggingface_hub"
                or name.startswith("huggingface_hub.")
            ):
                sys.modules.pop(name, None)
        for key in tuple(sys.path_importer_cache):
            text = os.fspath(key)
            if any(marker in text for marker in markers):
                sys.path_importer_cache.pop(key, None)
        importlib.invalidate_caches()


_sanitize_runtime_pydeps_imports()

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("MYVLA_DISABLE_TORCH_COMPILE", "1")

from rpc_pickle import serve_forever
from policy_prompting import (
    compose_bimanual_low_level_prompt,
    compose_low_level_prompt,
    fold_control_phase_name,
    infer_fold_control_phase,
    merge_subtask_with_phase,
)


def _add_myvla_to_syspath() -> Path:
    myvla_root = Path(__file__).resolve().parents[1]
    if os.fspath(myvla_root) not in sys.path:
        sys.path.insert(0, os.fspath(myvla_root))
    return myvla_root


def _load_policy_runtime_spec(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    config_path = checkpoint_path / "config.json"
    run_meta_path = checkpoint_path / "run_meta.json"

    config: dict[str, Any] = {}
    run_meta: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if run_meta_path.is_file():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    model_action_dim = int(
        run_meta.get(
            "model_action_dim",
            config.get("action_dim", 32),
        )
    )
    active_action_dim = int(
        run_meta.get(
            "active_action_dim",
            60 if model_action_dim > 32 else model_action_dim,
        )
    )
    if model_action_dim > 32 or active_action_dim > 32:
        policy_io_mode = "dex_bimanual"
    else:
        policy_io_mode = "droid"
    return {
        "policy_io_mode": str(policy_io_mode),
        "active_action_dim": int(active_action_dim),
        "model_action_dim": int(model_action_dim),
        "config_path": os.fspath(config_path),
        "run_meta_path": os.fspath(run_meta_path),
    }


def _ensure_uint8_hwc_rgb(image: Any) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    if x.ndim == 3 and x.shape[-1] == 4:
        x = x[..., :3]
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (HWC), got shape: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32)
        if xf.size and float(xf.max()) > 2.0:
            xf = xf / 255.0
        xf = np.clip(xf, 0.0, 1.0)
        x = (xf * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def _write_state_file(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    state_path = Path(path).expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def _resize_to_height(image: Any, *, target_h: int) -> np.ndarray:
    rgb = _ensure_uint8_hwc_rgb(image)
    if rgb.shape[0] == int(target_h):
        return rgb
    width = max(1, int(round(rgb.shape[1] * (float(target_h) / float(rgb.shape[0])))))
    return np.asarray(Image.fromarray(rgb).resize((width, int(target_h)), Image.Resampling.BILINEAR))


def _stack_mean_abs_diff(stack: Any) -> tuple[float, float]:
    video = np.asarray(stack)
    if video.ndim != 4 or video.shape[0] < 2:
        return 0.0, 0.0
    diffs = [
        float(np.mean(np.abs(video[index].astype(np.float32) - video[index - 1].astype(np.float32))))
        for index in range(1, video.shape[0])
    ]
    return float(np.mean(diffs)), float(diffs[-1])


def _build_hl_video_input(*, overview_stack: Any, cloth_stack: Any) -> np.ndarray:
    overview = np.asarray(overview_stack)
    cloth = np.asarray(cloth_stack)
    if overview.ndim == 3:
        overview = overview[None, ...]
    if cloth.ndim == 3:
        cloth = cloth[None, ...]
    if overview.ndim != 4 or cloth.ndim != 4:
        raise ValueError(
            f"Expected overview/cloth stacks with shape [T,H,W,C], got {overview.shape} and {cloth.shape}"
        )
    count = min(int(overview.shape[0]), int(cloth.shape[0]))
    overview = overview[-count:]
    cloth = cloth[-count:]
    frames: list[np.ndarray] = []
    for index in range(count):
        overview_frame = _ensure_uint8_hwc_rgb(overview[index])
        cloth_frame = _ensure_uint8_hwc_rgb(cloth[index])
        target_h = min(int(overview_frame.shape[0]), int(cloth_frame.shape[0]))
        frames.append(
            np.concatenate(
                [
                    _resize_to_height(overview_frame, target_h=target_h),
                    _resize_to_height(cloth_frame, target_h=target_h),
                ],
                axis=1,
            )
        )
    return np.stack(frames, axis=0)


def _completion_metrics(*, cloth_stack: Any, overview_stack: Any) -> dict[str, Any]:
    cloth_mean, cloth_last = _stack_mean_abs_diff(cloth_stack)
    overview_mean, overview_last = _stack_mean_abs_diff(overview_stack)
    frame_count = int(np.asarray(cloth_stack).shape[0]) if np.asarray(cloth_stack).ndim == 4 else 1
    return {
        "frame_count": frame_count,
        "cloth_tail_mean_delta": cloth_mean,
        "cloth_last_delta": cloth_last,
        "overview_tail_mean_delta": overview_mean,
        "overview_last_delta": overview_last,
    }


class FrameStacker:
    def __init__(self, *, window: int) -> None:
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self._buf: dict[str, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self._buf.clear()

    def push(self, key: str, frame: Any) -> None:
        f = _ensure_uint8_hwc_rgb(frame)
        if key not in self._buf:
            self._buf[key] = deque(maxlen=self.window)
        self._buf[key].append(f)

    def get(self, key: str) -> np.ndarray:
        if key not in self._buf or len(self._buf[key]) == 0:
            raise KeyError(f"No frames for key={key!r}")
        if self.window == 1:
            return self._buf[key][-1]
        frames = list(self._buf[key])
        if len(frames) < self.window:
            frames = [frames[0]] * (self.window - len(frames)) + frames
        return np.stack(frames, axis=0)  # [T,H,W,C]


class Pi05MemSession:
    def __init__(
        self,
        *,
        policy,
        hl,
        video_window: int,
        viz_base: Path,
        viz_name: str | None,
        server_meta: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.policy_io_mode = str(getattr(policy, "io_mode", "droid"))
        self.hl = hl
        self.video_window = int(video_window)
        self.stack_left = FrameStacker(window=self.video_window)
        self.stack_right = FrameStacker(window=self.video_window)
        self.stack_overview = FrameStacker(window=self.video_window)
        self.stack_cloth = FrameStacker(window=self.video_window)
        self.language_memory = ""
        self.goal = ""
        self.subtask = ""
        self.control_phase_key = "approach_left_sleeve"
        self.control_phase_name = fold_control_phase_name(self.control_phase_key)
        self.last_actions_both: np.ndarray | None = None

        from myvla_mem.viz import InferenceVizWriter

        meta = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "server": server_meta}
        self.writer = InferenceVizWriter.create(base_dir=viz_base, name=viz_name, meta=meta)

    def reset(self, *, goal: str, language_memory: str = "") -> None:
        self.goal = str(goal).strip()
        self.subtask = self.goal
        self.language_memory = str(language_memory or "")
        self.control_phase_key = "approach_left_sleeve"
        self.control_phase_name = fold_control_phase_name(self.control_phase_key)
        self.last_actions_both = None
        self.stack_left.reset()
        self.stack_right.reset()
        self.stack_overview.reset()
        self.stack_cloth.reset()
        if self.hl is not None and hasattr(self.hl, "reset"):
            self.hl.reset()

    def step(
        self,
        step: int,
        *,
        base_rgb: Any,
        cloth_rgb: Any | None = None,
        left_exterior_rgb: Any | None = None,
        right_exterior_rgb: Any | None = None,
        left_wrist_rgb: Any,
        right_wrist_rgb: Any,
        jL: Any,
        gL: Any,
        jR: Any,
        gR: Any,
        jL_full: Any | None = None,
        jR_full: Any | None = None,
        num_steps: int,
        phase_hint_name: str | None = None,
    ) -> dict[str, Any]:
        goal = self.goal
        prev_memory = self.language_memory

        base_u8 = _ensure_uint8_hwc_rgb(base_rgb)
        cloth_u8 = _ensure_uint8_hwc_rgb(cloth_rgb if cloth_rgb is not None else left_exterior_rgb if left_exterior_rgb is not None else base_rgb)
        left_ext_u8 = _ensure_uint8_hwc_rgb(left_exterior_rgb if left_exterior_rgb is not None else base_rgb)
        right_ext_u8 = _ensure_uint8_hwc_rgb(right_exterior_rgb if right_exterior_rgb is not None else base_rgb)
        left_u8 = _ensure_uint8_hwc_rgb(left_wrist_rgb)
        right_u8 = _ensure_uint8_hwc_rgb(right_wrist_rgb)

        self.stack_left.push("base", left_ext_u8)
        self.stack_left.push("wrist", left_u8)
        self.stack_right.push("base", right_ext_u8)
        self.stack_right.push("wrist", right_u8)
        self.stack_overview.push("overview", base_u8)
        self.stack_cloth.push("cloth", cloth_u8)

        subtask = goal
        hl_raw_text = None
        structured_state: dict[str, Any] = {}
        retrieved_semantic_summary = ""
        retrieved_visual_summary = ""
        retrieved_semantic_hint = ""
        retrieved_visual_hint = ""
        pcmb_debug: dict[str, Any] = {}
        phase_hint_name = str(phase_hint_name or self.control_phase_name or "").strip()
        completion_metrics = _completion_metrics(
            cloth_stack=self.stack_cloth.get("cloth"),
            overview_stack=self.stack_overview.get("overview"),
        )
        done = False
        completion_reason = ""
        completion_score = None
        if self.hl is not None:
            hl_result = self.hl.update(
                goal=goal,
                prev_memory=self.language_memory,
                image=_build_hl_video_input(
                    overview_stack=self.stack_overview.get("overview"),
                    cloth_stack=self.stack_cloth.get("cloth"),
                ),
                phase_name=phase_hint_name,
                video_metrics=completion_metrics,
                step=int(step),
            )
            self.language_memory = hl_result.memory
            subtask = hl_result.subtask
            hl_raw_text = hl_result.raw_text
            structured_state = dict(hl_result.structured_state)
            retrieved_semantic_summary = str(hl_result.retrieved_semantic_summary)
            retrieved_visual_summary = str(hl_result.retrieved_visual_summary)
            retrieved_semantic_hint = str(hl_result.retrieved_semantic_hint)
            retrieved_visual_hint = str(hl_result.retrieved_visual_hint)
            pcmb_debug = dict(hl_result.pcmb_debug)
            completion_score = hl_result.completion_score
            completion_reason = str(hl_result.completion_reason or "")
            done = bool(hl_result.done)

            cloth_is_stable = float(completion_metrics["cloth_tail_mean_delta"]) <= 2.0
            if done and not cloth_is_stable:
                done = False
                completion_reason = (
                    completion_reason + "; waiting for shirt motion to settle"
                    if completion_reason
                    else "waiting for shirt motion to settle"
                )
            elif (not done) and completion_score is not None and completion_score >= 97.0 and cloth_is_stable:
                done = True
                completion_reason = "high completion score with stable recent cloth video"
        control_subtask = merge_subtask_with_phase(goal=goal, subtask=subtask, phase_name=phase_hint_name)
        control_phase_key = infer_fold_control_phase(
            subtask=control_subtask,
            structured_state=structured_state,
            fallback_phase_name=phase_hint_name,
        )
        if done:
            control_phase_key = "inspect_finish"
        control_phase_name = fold_control_phase_name(control_phase_key)
        prev_control_phase_key = str(self.control_phase_key)
        prev_control_phase_name = str(self.control_phase_name)
        self.control_phase_key = str(control_phase_key)
        self.control_phase_name = str(control_phase_name)

        low_level_prompt_left = compose_low_level_prompt(
            goal=goal,
            subtask=control_subtask,
            language_memory=self.language_memory,
            arm_side="left",
            phase_name=control_phase_name,
            structured_state_summary="; ".join(f"{k}={v}" for k, v in structured_state.items() if v),
            retrieved_semantic_hint=retrieved_semantic_hint,
            retrieved_visual_hint=retrieved_visual_hint,
        )
        low_level_prompt_right = compose_low_level_prompt(
            goal=goal,
            subtask=control_subtask,
            language_memory=self.language_memory,
            arm_side="right",
            phase_name=control_phase_name,
            structured_state_summary="; ".join(f"{k}={v}" for k, v in structured_state.items() if v),
            retrieved_semantic_hint=retrieved_semantic_hint,
            retrieved_visual_hint=retrieved_visual_hint,
        )
        goal_only_prompt_mode = "No stage label or subtask label is provided" in low_level_prompt_left
        reported_subtask = goal if goal_only_prompt_mode else control_subtask
        self.subtask = reported_subtask
        low_level_prompt = low_level_prompt_left

        if self.policy_io_mode == "dex_bimanual":
            prompt_bimanual = compose_bimanual_low_level_prompt(
                goal=goal,
                subtask=control_subtask,
                language_memory=self.language_memory,
                phase_name=control_phase_name,
                structured_state_summary="; ".join(f"{k}={v}" for k, v in structured_state.items() if v),
                retrieved_semantic_hint=retrieved_semantic_hint,
                retrieved_visual_hint=retrieved_visual_hint,
            )
            low_level_prompt = prompt_bimanual
            ex_bimanual = {
                "observation/exterior_image_1_left": self.stack_left.get("base"),
                "observation/exterior_image_1_right": self.stack_right.get("base"),
                "observation/wrist_image_left": self.stack_left.get("wrist"),
                "observation/wrist_image_right": self.stack_right.get("wrist"),
                "observation/joint_position_left": np.asarray(jL_full if jL_full is not None else jL, dtype=np.float32),
                "observation/joint_position_right": np.asarray(jR_full if jR_full is not None else jR, dtype=np.float32),
                "prompt": prompt_bimanual,
            }
            out_bimanual = self.policy.infer(ex_bimanual, num_steps=int(num_steps))
            actions_both = np.asarray(out_bimanual["actions"], dtype=np.float32)
            split = max(1, int(actions_both.shape[1] // 2))
            actions_left = actions_both[:, :split]
            actions_right = actions_both[:, split : split * 2]
        else:
            ex_left = {
                "observation/exterior_image_1_left": self.stack_left.get("base"),
                "observation/wrist_image_left": self.stack_left.get("wrist"),
                "observation/joint_position": np.asarray(jL, dtype=np.float32),
                "observation/gripper_position": np.asarray(gL, dtype=np.float32),
                "prompt": low_level_prompt_left,
            }
            ex_right = {
                "observation/exterior_image_1_left": self.stack_right.get("base"),
                "observation/wrist_image_left": self.stack_right.get("wrist"),
                "observation/joint_position": np.asarray(jR, dtype=np.float32),
                "observation/gripper_position": np.asarray(gR, dtype=np.float32),
                "prompt": low_level_prompt_right,
            }

            out_left = self.policy.infer(ex_left, num_steps=int(num_steps))
            out_right = self.policy.infer(ex_right, num_steps=int(num_steps))

            actions_left = np.asarray(out_left["actions"], dtype=np.float32)
            actions_right = np.asarray(out_right["actions"], dtype=np.float32)
            actions_both = np.concatenate([actions_left, actions_right], axis=1)
        self.last_actions_both = actions_both

        self.writer.add_step(
            int(step),
            goal=goal,
            low_level_prompt=low_level_prompt,
            prev_memory=prev_memory,
            language_memory=self.language_memory,
            subtask=reported_subtask,
            hl_raw_text=hl_raw_text,
            structured_state=structured_state,
            retrieved_semantic_summary=retrieved_semantic_summary,
            retrieved_visual_summary=retrieved_visual_summary,
            pcmb_debug=pcmb_debug,
            images={
                "overview": base_u8,
                "left_exterior": self.stack_left.get("base"),
                "right_exterior": self.stack_right.get("base"),
                "left_wrist": self.stack_left.get("wrist"),
                "right_wrist": self.stack_right.get("wrist"),
            },
            actions=actions_both,
        )

        return {
            "ok": True,
            "hl_enabled": bool(self.hl is not None),
            "actions_left": actions_left,
            "actions_right": actions_right,
            "actions_both": actions_both,
            "language_memory": self.language_memory,
            "subtask": reported_subtask,
            "prev_control_phase_key": prev_control_phase_key,
            "prev_control_phase_name": prev_control_phase_name,
            "phase_hint_name": phase_hint_name,
            "control_phase_key": control_phase_key,
            "control_phase_name": control_phase_name,
            "policy_io_mode": self.policy_io_mode,
            "hl_raw_text": hl_raw_text,
            "structured_state": structured_state,
            "retrieved_semantic_summary": retrieved_semantic_summary,
            "retrieved_visual_summary": retrieved_visual_summary,
            "retrieved_semantic_hint": retrieved_semantic_hint,
            "retrieved_visual_hint": retrieved_visual_hint,
            "pcmb_debug": pcmb_debug,
            "active_action_dim": int(getattr(self.policy, "active_action_dim", actions_both.shape[1] if actions_both.ndim == 2 else 0)),
            "model_action_dim": int(getattr(self.policy, "model_action_dim", actions_both.shape[1] if actions_both.ndim == 2 else 0)),
            "done": bool(done),
            "completion_reason": completion_reason,
            "completion_score": completion_score,
            "completion_metrics": completion_metrics,
            "low_level_prompt": low_level_prompt,
            "low_level_prompt_left": low_level_prompt_left,
            "low_level_prompt_right": low_level_prompt_right,
            "viz_run_dir": os.fspath(self.writer.run_dir),
        }

    def finalize(self) -> dict[str, Any]:
        self.writer.finalize(final_actions=self.last_actions_both if self.last_actions_both is not None else None)
        (self.writer.run_dir / "final_language_memory.txt").write_text(self.language_memory, encoding="utf-8")
        (self.writer.run_dir / "summary.json").write_text(
            json.dumps(
                {"goal": self.goal, "final_subtask": self.subtask, "final_memory_len": len(self.language_memory)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"ok": True, "viz_run_dir": os.fspath(self.writer.run_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="myVLA pi0.5(+memory) RPC server for Isaac Sim (pickle over TCP).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout_s", type=float, default=300.0)

    parser.add_argument("--checkpoint_dir", default=str(Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"))
    parser.add_argument("--tokenizer_model", default="", help="Optional paligemma_tokenizer.model path")
    parser.add_argument("--device", default="", help="Torch device for pi0.5 (e.g. cuda:0/cpu). Default: auto")
    parser.add_argument("--num_steps", type=int, default=4, help="Default num_steps if client omits it")
    parser.add_argument("--video_window", type=int, default=4)
    parser.add_argument("--policy_io_mode", default="auto", choices=("auto", "droid", "dex_bimanual"))
    parser.add_argument("--active_action_dim", type=int, default=0)
    parser.add_argument("--model_action_dim", type=int, default=0)

    parser.add_argument("--hl_vlm_dir", default="", help="Pretrained VLM directory/id (optional)")
    parser.add_argument("--hl_device", default="cpu")
    parser.add_argument("--hl_dtype", default="bfloat16")
    parser.add_argument("--hl_revision", default="")
    parser.add_argument("--hl_max_new_tokens", type=int, default=64)
    parser.add_argument("--hl_temperature", type=float, default=0.0)

    parser.add_argument("--viz_dir", default="", help="Output directory (default: myVLA/isaac_sim_viz/<timestamp>)")
    parser.add_argument("--viz_name", default="", help="Run folder name (optional)")
    parser.add_argument("--state_file", default="", help="Optional runtime state file for warm/reusable RPC.")
    args = parser.parse_args()
    print("[rpc] parsed_args", flush=True)
    _write_state_file(
        str(args.state_file).strip(),
        {
            "pid": int(os.getpid()),
            "ready": False,
            "status": "bootstrapping",
            "phase": "parsed_args",
            "args": vars(args),
        },
    )

    print("[rpc] add_myvla_to_syspath:start", flush=True)
    myvla_root = _add_myvla_to_syspath()
    print(f"[rpc] add_myvla_to_syspath:done myvla_root={myvla_root}", flush=True)
    _write_state_file(
        str(args.state_file).strip(),
        {
            "pid": int(os.getpid()),
            "ready": False,
            "status": "bootstrapping",
            "phase": "syspath_ready",
            "myvla_root": os.fspath(myvla_root),
            "args": vars(args),
        },
    )

    print("[rpc] import_pi05_policy:start", flush=True)
    from myvla_pi05.policy import Pi05DexBimanualPolicy, Pi05DroidPolicy
    print("[rpc] import_pi05_policy:done", flush=True)

    print(f"[rpc] construct_policy:start checkpoint_dir={args.checkpoint_dir}", flush=True)
    runtime_spec = _load_policy_runtime_spec(args.checkpoint_dir)
    requested_io_mode = str(args.policy_io_mode).strip() or "auto"
    policy_io_mode = str(runtime_spec["policy_io_mode"]) if requested_io_mode == "auto" else requested_io_mode
    active_action_dim = int(args.active_action_dim) if int(args.active_action_dim) > 0 else int(runtime_spec["active_action_dim"])
    model_action_dim = int(args.model_action_dim) if int(args.model_action_dim) > 0 else int(runtime_spec["model_action_dim"])
    print(
        "[rpc] policy_runtime_spec="
        + json.dumps(
            {
                "policy_io_mode": policy_io_mode,
                "active_action_dim": active_action_dim,
                "model_action_dim": model_action_dim,
                "config_path": runtime_spec["config_path"],
                "run_meta_path": runtime_spec["run_meta_path"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if policy_io_mode == "dex_bimanual":
        policy = Pi05DexBimanualPolicy(
            args.checkpoint_dir,
            device=str(args.device).strip() or None,
            tokenizer_model=str(args.tokenizer_model).strip() or None,
            active_action_dim=int(active_action_dim),
            model_action_dim=int(model_action_dim),
        )
    else:
        policy = Pi05DroidPolicy(
            args.checkpoint_dir,
            device=str(args.device).strip() or None,
            tokenizer_model=str(args.tokenizer_model).strip() or None,
        )
    print("[rpc] construct_policy:done", flush=True)
    _write_state_file(
        str(args.state_file).strip(),
        {
            "pid": int(os.getpid()),
            "ready": False,
            "status": "bootstrapping",
            "phase": "policy_ready",
            "checkpoint_dir": str(args.checkpoint_dir),
            "device": str(args.device).strip(),
            "args": vars(args),
        },
    )

    hl = None
    if str(args.hl_vlm_dir).strip():
        print("[rpc] import_hl_processor:start", flush=True)
        from myvla_mem.long_term import PretrainedVlmLongTermMemoryProcessor
        print("[rpc] import_hl_processor:done", flush=True)

        print(f"[rpc] construct_hl:start hl_vlm_dir={args.hl_vlm_dir}", flush=True)
        hl = PretrainedVlmLongTermMemoryProcessor(
            str(args.hl_vlm_dir).strip(),
            device=str(args.hl_device).strip() or None,
            dtype=str(args.hl_dtype),
            revision=str(args.hl_revision).strip() or None,
            max_new_tokens=int(args.hl_max_new_tokens),
            temperature=float(args.hl_temperature),
        )
        print("[rpc] construct_hl:done", flush=True)

    viz_base = (
        Path(str(args.viz_dir)).expanduser().resolve()
        if str(args.viz_dir).strip()
        else (myvla_root / "isaac_sim_viz")
    )
    server_meta = {"args": vars(args)}
    print(f"[rpc] construct_session:start viz_base={viz_base}", flush=True)
    session = Pi05MemSession(
        policy=policy,
        hl=hl,
        video_window=int(args.video_window),
        viz_base=viz_base,
        viz_name=str(args.viz_name).strip() or None,
        server_meta=server_meta,
    )
    print("[rpc] construct_session:done", flush=True)
    state_file = str(args.state_file).strip()
    server_status = {
        "pid": int(os.getpid()),
        "host": str(args.host),
        "port": int(args.port),
        "ready": True,
        "status": "ready",
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "hl_vlm_dir": str(args.hl_vlm_dir).strip(),
        "device": str(args.device).strip(),
        "hl_device": str(args.hl_device).strip(),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip(),
        "video_window": int(args.video_window),
        "viz_run_dir": os.fspath(session.writer.run_dir),
        "args": vars(args),
    }
    _write_state_file(state_file, server_status)
    print(f"[rpc] ready state_file={state_file}", flush=True)

    def _mark_stopped() -> None:
        if not state_file:
            return
        stopped = dict(server_status)
        stopped["ready"] = False
        stopped["status"] = "stopped"
        stopped["stopped_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        _write_state_file(state_file, stopped)

    atexit.register(_mark_stopped)
    print("[rpc] entering_serve_forever", flush=True)

    def handler(msg: dict[str, Any]) -> dict[str, Any]:
        cmd = str(msg.get("cmd", "")).lower()
        if cmd in ("ping", "status", "health"):
            return {
                "ok": True,
                "ready": True,
                "pid": int(os.getpid()),
                "host": str(args.host),
                "port": int(args.port),
                "viz_run_dir": os.fspath(session.writer.run_dir),
                "state_file": state_file,
            }
        if cmd in ("reset", "start"):
            session.reset(goal=str(msg.get("goal", "")).strip(), language_memory=str(msg.get("language_memory", "") or ""))
            return {"ok": True, "viz_run_dir": os.fspath(session.writer.run_dir)}
        if cmd in ("step", "infer"):
            return session.step(
                int(msg.get("step", 0)),
                base_rgb=msg["base_rgb"],
                cloth_rgb=msg.get("cloth_rgb"),
                left_exterior_rgb=msg.get("left_exterior_rgb"),
                right_exterior_rgb=msg.get("right_exterior_rgb"),
                left_wrist_rgb=msg["left_wrist_rgb"],
                right_wrist_rgb=msg["right_wrist_rgb"],
                jL=msg["jL"],
                gL=msg["gL"],
                jR=msg["jR"],
                gR=msg["gR"],
                jL_full=msg.get("jL_full"),
                jR_full=msg.get("jR_full"),
                num_steps=int(msg.get("num_steps", int(args.num_steps))),
                phase_hint_name=str(msg.get("phase_hint_name", "") or ""),
            )
        if cmd in ("finalize",):
            return session.finalize()
        if cmd in ("close", "quit", "exit"):
            session.finalize()
            _mark_stopped()
            return {"ok": True, "viz_run_dir": os.fspath(session.writer.run_dir)}
        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

    print(f"[rpc] listening on {args.host}:{int(args.port)}")
    print(f"[rpc] viz run dir: {session.writer.run_dir}")
    serve_forever(host=str(args.host), port=int(args.port), handler=handler, timeout_s=float(args.timeout_s))


if __name__ == "__main__":
    main()
