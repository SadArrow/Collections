from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from policy_prompting import compose_low_level_prompt, fold_phase_name, merge_subtask_with_phase


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _add_myvla_to_syspath() -> Path:
    # myVLA/isaac_sim/fold_shirt_dual_arm_mem_pi05.py -> myVLA
    myvla_root = Path(__file__).resolve().parents[1]
    if os.fspath(myvla_root) not in sys.path:
        sys.path.insert(0, os.fspath(myvla_root))
    return myvla_root


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_rpc(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with socket.create_connection((str(host), int(port)), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for RPC server at {host}:{port}")


def _load_rpc_state(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        state = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _rpc_ping(host: str, port: int, timeout_s: float) -> dict[str, Any] | None:
    from rpc_pickle import RpcClient

    rpc = RpcClient(host=str(host), port=int(port), timeout_s=float(timeout_s))
    try:
        return rpc.call({"cmd": "ping"})
    except Exception:
        return None
    finally:
        rpc.close()


def _wait_for_rpc_ready(host: str, port: int, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    last_pong: dict[str, Any] | None = None
    while time.time() < deadline:
        last_pong = _rpc_ping(host, port, min(10.0, float(timeout_s)))
        if last_pong and last_pong.get("ready"):
            return last_pong
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for ready RPC server at {host}:{port}; last_pong={last_pong}")


def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _spawn_detached_windows(cmd: list[str], *, cwd: Path, log_path: str | None = None) -> None:
    file_path = _ps_quote(str(cmd[0]))
    arg_list = ",".join(_ps_quote(str(x)) for x in cmd[1:])
    launcher = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"Start-Process -FilePath {file_path} -WorkingDirectory {_ps_quote(os.fspath(cwd))} "
        f"-WindowStyle Hidden -ArgumentList {arg_list}",
    ]
    subprocess.run(launcher, check=True, cwd=os.fspath(cwd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603


def _normalize(v: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.asarray([w, x, y, z], dtype=np.float32)
    return _normalize(q)


def _look_at_quat_wxyz(*, camera_pos: np.ndarray, target_pos: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """Quaternion for a USD camera that looks at target (camera -Z forward, +Y up; quat is [w,x,y,z])."""
    if world_up is None:
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    forward = _normalize(target_pos - camera_pos)  # world direction to target
    z_axis = -forward  # USD camera looks down -Z
    up = _normalize(world_up)
    x_axis = np.cross(up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))

    # Columns are local basis vectors expressed in world frame: [x, y, z]
    R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return _rotmat_to_quat_wxyz(R)


def _yaw_quat_wxyz(yaw_deg: float) -> np.ndarray:
    """Yaw-only quaternion (about +Z) in [w,x,y,z]."""
    half = 0.5 * math.radians(float(yaw_deg))
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _as_float32_quat(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    return _normalize(q)


def _quat_conjugate_wxyz(quat: Any) -> np.ndarray:
    q = _as_float32_quat(quat)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_multiply_wxyz(q1: Any, q2: Any) -> np.ndarray:
    a = _as_float32_quat(q1)
    b = _as_float32_quat(q2)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_rotate_vec_wxyz(quat: Any, vec: Any) -> np.ndarray:
    q = _as_float32_quat(quat)
    v = np.asarray(vec, dtype=np.float32).reshape(3)
    qv = np.asarray([0.0, v[0], v[1], v[2]], dtype=np.float32)
    rotated = _quat_multiply_wxyz(_quat_multiply_wxyz(q, qv), _quat_conjugate_wxyz(q))
    return rotated[1:].astype(np.float32)


def _quat_mul_batch_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def _quat_conjugate_batch_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[:, :1], -q[:, 1:]], axis=-1).astype(np.float32)


def _differential_inverse_kinematics(
    *,
    jacobian_end_effector: np.ndarray,
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    goal_position: np.ndarray,
    goal_orientation: np.ndarray | None = None,
    damping: float = 0.05,
    scale: float = 1.0,
) -> np.ndarray:
    goal_orientation = current_orientation if goal_orientation is None else goal_orientation
    q_err = _quat_mul_batch_wxyz(goal_orientation, _quat_conjugate_batch_wxyz(current_orientation))
    error = np.expand_dims(
        np.concatenate(
            [
                goal_position - current_position,
                q_err[:, 1:] * np.sign(np.where(np.abs(q_err[:, :1]) < 1e-6, 1.0, q_err[:, :1])),
            ],
            axis=-1,
        ),
        axis=2,
    )
    jac_t = np.swapaxes(jacobian_end_effector, 1, 2)
    lmbda = np.eye(jacobian_end_effector.shape[1], dtype=np.float32) * float(damping * damping)
    dq = scale * (jac_t @ np.linalg.inv(jacobian_end_effector @ jac_t + lmbda) @ error).squeeze(-1)
    return dq.astype(np.float32)


class FrameStacker:
    def __init__(self, *, window: int) -> None:
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self._buf: dict[str, deque[np.ndarray]] = {}

    def push(self, key: str, frame: Any) -> None:
        x = np.asarray(frame)
        if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
            x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
        if x.ndim == 3 and x.shape[-1] == 4:
            x = x[..., :3]
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape={x.shape}")
        if np.issubdtype(x.dtype, np.floating):
            xf = x.astype(np.float32)
            if xf.size and float(xf.max()) > 2.0:
                xf = xf / 255.0
            xf = np.clip(xf, 0.0, 1.0)
            x = (xf * 255.0).astype(np.uint8)
        elif x.dtype != np.uint8:
            x = x.astype(np.uint8)
        if key not in self._buf:
            self._buf[key] = deque(maxlen=self.window)
        self._buf[key].append(x)

    def get(self, key: str) -> np.ndarray:
        if self.window == 1:
            if key not in self._buf or len(self._buf[key]) == 0:
                raise KeyError(f"No frames for key={key!r}")
            return self._buf[key][-1]

        if key not in self._buf or len(self._buf[key]) == 0:
            raise KeyError(f"No frames for key={key!r}")
        frames = list(self._buf[key])
        if len(frames) < self.window:
            frames = [frames[0]] * (self.window - len(frames)) + frames
        return np.stack(frames, axis=0)  # [T,H,W,C]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _coerce_camera_rgb(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    x = np.asarray(frame)
    if x.ndim == 0 or x.size == 0:
        return None
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] == 4:
        x = x[..., :3]
    if x.ndim != 3 or x.shape[-1] != 3:
        return None
    if np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32)
        if xf.size and float(xf.max()) > 2.0:
            xf = xf / 255.0
        xf = np.clip(xf, 0.0, 1.0)
        x = (xf * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def _read_camera_rgb(camera: Any, *, world: Any, max_attempts: int = 8) -> np.ndarray:
    for attempt in range(int(max_attempts)):
        frame = _coerce_camera_rgb(camera.get_rgb())
        if frame is None:
            try:
                frame = _coerce_camera_rgb(camera.get_rgba())
            except Exception:  # noqa: BLE001
                frame = None
        if frame is None:
            try:
                current = camera.get_current_frame()
            except Exception:  # noqa: BLE001
                current = None
            if isinstance(current, dict):
                frame = _coerce_camera_rgb(current.get("rgb"))
                if frame is None:
                    frame = _coerce_camera_rgb(current.get("rgba"))
        if frame is not None:
            return frame
        if attempt + 1 < int(max_attempts):
            world.step(render=True)
    raise RuntimeError(f"Failed to read RGB frame from camera after {int(max_attempts)} attempts: {camera}")


def _setup_scene(*, simulation_app, backend: str, device: str, table_z: float) -> dict[str, Any]:
    import omni.usd
    import omni.kit.commands
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.stage import get_stage_units, is_stage_loading
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    def set_if(obj: Any, name: str, value: Any) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    # ------------------------------------------------------------------
    # Offline-friendly robot loading:
    # Avoid `get_assets_root_path()` (which may require network access) by importing a Franka URDF
    # from the local Isaac Sim distribution (CuRobo content) and caching it as a USD file.
    # ------------------------------------------------------------------
    myvla_root = Path(__file__).resolve().parents[1]
    usd_cache_dir = myvla_root / "isaac_sim_assets_cache"
    usd_cache_dir.mkdir(parents=True, exist_ok=True)
    franka_usd_cache = usd_cache_dir / "franka_panda.usd"

    isaac_root = Path(os.environ.get("ISAAC_PATH") or "").expanduser()
    if not isaac_root or not isaac_root.exists():
        isaac_root = myvla_root.parent / "issac-sim"

    franka_candidates = [
        isaac_root
        / "curobo"
        / "src"
        / "curobo"
        / "content"
        / "assets"
        / "robot"
        / "franka_description"
        / "franka_panda.urdf",
        isaac_root
        / "exts"
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "franka_description"
        / "robots"
        / "panda_arm_hand.urdf",
        isaac_root
        / "exts"
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "franka_description"
        / "robots"
        / "panda_arm.urdf",
    ]
    franka_urdf = next((candidate for candidate in franka_candidates if candidate.exists()), None)
    if franka_urdf is None:
        searched = "\n".join(os.fspath(candidate) for candidate in franka_candidates)
        raise FileNotFoundError(f"Franka URDF not found. Searched:\n{searched}")

    meta_path = franka_usd_cache.with_suffix(".meta.json")
    regen_cache = (not franka_usd_cache.exists()) or (not meta_path.exists())
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
        if str(meta.get("urdf_path", "")) != os.fspath(franka_urdf.resolve()):
            regen_cache = True

    if regen_cache:
        import shutil

        # Clear stale cached files from older URDFs/configs.
        if franka_usd_cache.exists():
            franka_usd_cache.unlink()
        cfg_dir = usd_cache_dir / "configuration"
        if cfg_dir.exists():
            shutil.rmtree(cfg_dir, ignore_errors=True)

        enable_extension("isaacsim.asset.importer.urdf")
        simulation_app.update()

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")

        set_if(import_config, "merge_fixed_joints", False)
        set_if(import_config, "convex_decomp", False)
        set_if(import_config, "import_inertia_tensor", True)
        set_if(import_config, "fix_base", True)
        set_if(import_config, "make_default_prim", True)
        set_if(import_config, "create_physics_scene", False)
        set_if(import_config, "distance_scale", 1.0)

        status, _prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=os.fspath(franka_urdf),
            import_config=import_config,
            dest_path=os.fspath(franka_usd_cache),
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError("URDFParseAndImportFile failed for Franka URDF")

        meta_path.write_text(json.dumps({"urdf_path": os.fspath(franka_urdf.resolve())}, indent=2) + "\n", encoding="utf-8")

    stage = omni.usd.get_context().get_stage()
    # Use meters consistently for all scene geometry/poses.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage_units = float(get_stage_units())

    my_world = World(stage_units_in_meters=stage_units, backend=str(backend), device=str(device))

    # Procedural ground (avoid default ground plane assets which may require network access).
    ground = FixedCuboid(
        name="ground",
        prim_path="/World/Ground",
        position=np.array([0.0, 0.0, -0.05], dtype=np.float32),
        scale=np.array([20.0, 20.0, 0.1], dtype=np.float32),
        color=np.array([0.07, 0.12, 0.22], dtype=np.float32),
    )
    my_world.scene.add(ground)

    # Low-contrast omnidirectional lighting: dim dome + several soft sphere lights.
    fill = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    fill.CreateIntensityAttr(160)
    fill.CreateExposureAttr(0.0)
    fill.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    fill.CreateTextureFormatAttr("latlong")

    sphere_specs = [
        ("/World/Lights/FrontLeft", np.array([2.3, 1.8, 2.6], dtype=np.float32), 24000.0, Gf.Vec3f(0.90, 0.93, 1.0)),
        ("/World/Lights/FrontRight", np.array([2.3, -1.8, 2.6], dtype=np.float32), 24000.0, Gf.Vec3f(0.90, 0.93, 1.0)),
        ("/World/Lights/BackLeft", np.array([-1.6, 1.8, 2.4], dtype=np.float32), 18000.0, Gf.Vec3f(0.70, 0.78, 1.0)),
        ("/World/Lights/BackRight", np.array([-1.6, -1.8, 2.4], dtype=np.float32), 18000.0, Gf.Vec3f(0.70, 0.78, 1.0)),
    ]
    for light_path, light_pos, intensity, color in sphere_specs:
        sphere = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
        sphere.CreateIntensityAttr(float(intensity))
        sphere.CreateRadiusAttr(0.65)
        sphere.CreateExposureAttr(0.0)
        sphere.CreateColorAttr(color)
        light_xf = UsdGeom.Xformable(sphere.GetPrim())
        light_xf.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in light_pos]))

    # Simple colored backdrop (non-network, makes visuals easier to debug).
    backdrop = FixedCuboid(
        name="backdrop",
        prim_path="/World/Backdrop",
        position=np.array([-22.0, 0.0, 3.0], dtype=np.float32),
        scale=np.array([0.05, 40.0, 16.0], dtype=np.float32),
        color=np.array([0.03, 0.08, 0.18], dtype=np.float32),
    )
    my_world.scene.add(backdrop)

    # Table (static)
    table = FixedCuboid(
        name="table",
        prim_path="/World/Table",
        position=np.array([0.6, 0.0, float(table_z)], dtype=np.float32),
        scale=np.array([1.0, 1.4, 0.05], dtype=np.float32),
        color=np.array([0.26, 0.18, 0.12], dtype=np.float32),
    )
    my_world.scene.add(table)
    floor_pattern_specs = [
        ("FloorStripeA", [0.5, -3.0, -0.044], [7.0, 0.34, 0.012], [0.10, 0.18, 0.32]),
        ("FloorStripeB", [0.5, -1.1, -0.043], [7.0, 0.22, 0.014], [0.05, 0.11, 0.22]),
        ("FloorStripeC", [0.5, 1.0, -0.044], [7.0, 0.34, 0.012], [0.10, 0.18, 0.32]),
        ("FloorStripeD", [0.5, 3.0, -0.043], [7.0, 0.22, 0.014], [0.05, 0.11, 0.22]),
    ]
    for name, position, scale, color in floor_pattern_specs:
        my_world.scene.add(
            FixedCuboid(
                name=name,
                prim_path=f"/World/{name}",
                position=np.asarray(position, dtype=np.float32),
                scale=np.asarray(scale, dtype=np.float32),
                color=np.asarray(color, dtype=np.float32),
            )
        )
    table_pattern_specs = [
        ("TableStripeCenter", [0.6, 0.0, float(table_z) + 0.026], [0.95, 0.16, 0.008], [0.45, 0.32, 0.22]),
        ("TableStripeLeft", [0.6, 0.42, float(table_z) + 0.026], [0.90, 0.08, 0.006], [0.17, 0.10, 0.07]),
        ("TableStripeRight", [0.6, -0.42, float(table_z) + 0.026], [0.90, 0.08, 0.006], [0.17, 0.10, 0.07]),
    ]
    for name, position, scale, color in table_pattern_specs:
        my_world.scene.add(
            FixedCuboid(
                name=name,
                prim_path=f"/World/{name}",
                position=np.asarray(position, dtype=np.float32),
                scale=np.asarray(scale, dtype=np.float32),
                color=np.asarray(color, dtype=np.float32),
            )
        )

    # Two Franka arms
    add_reference_to_stage(usd_path=os.fspath(franka_usd_cache), prim_path="/World/ArmLeft")
    add_reference_to_stage(usd_path=os.fspath(franka_usd_cache), prim_path="/World/ArmRight")

    def set_xform_pose(*, prim_path: str, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
        prim = stage.GetPrimAtPath(str(prim_path))
        if not prim or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xf.GetOrderedXformOps():
            t = op.GetOpType()
            if t == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif t == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        if orient_op is None:
            orient_op = xf.AddOrientOp()
        p = np.asarray(pos_xyz, dtype=np.float32).reshape(3)
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        translate_op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
        orient_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))

    arm_left = Articulation(prim_paths_expr="/World/ArmLeft", name="arm_left")
    arm_right = Articulation(prim_paths_expr="/World/ArmRight", name="arm_right")
    my_world.scene.add(arm_left)
    my_world.scene.add(arm_right)

    # Default arm placement (on opposite sides of the table, facing inward).
    #
    # IMPORTANT: World.reset() will restore each prim to its default state. So we set the default
    # state here to prevent the two arms from snapping back to the same pose (overlapping).
    left_pos = np.array([[0.0, 0.9, 0.0]], dtype=np.float32)
    right_pos = np.array([[0.0, -0.9, 0.0]], dtype=np.float32)
    left_ori = _yaw_quat_wxyz(-90.0)[None, :]  # face -Y (towards table center)
    right_ori = _yaw_quat_wxyz(+90.0)[None, :]  # face +Y (towards table center)

    # Move the *container* prims too (more robust than relying on Articulation root discovery).
    set_xform_pose(prim_path="/World/ArmLeft", pos_xyz=left_pos[0], quat_wxyz=left_ori[0])
    set_xform_pose(prim_path="/World/ArmRight", pos_xyz=right_pos[0], quat_wxyz=right_ori[0])

    if str(backend).lower() == "torch":
        import torch

        dev = torch.device(str(device))
        left_pos_t = torch.as_tensor(left_pos, dtype=torch.float32, device=dev)
        right_pos_t = torch.as_tensor(right_pos, dtype=torch.float32, device=dev)
        left_ori_t = torch.as_tensor(left_ori, dtype=torch.float32, device=dev)
        right_ori_t = torch.as_tensor(right_ori, dtype=torch.float32, device=dev)

        arm_left.set_default_state(positions=left_pos_t, orientations=left_ori_t)
        arm_right.set_default_state(positions=right_pos_t, orientations=right_ori_t)
    else:
        arm_left.set_default_state(positions=left_pos, orientations=left_ori)
        arm_right.set_default_state(positions=right_pos, orientations=right_ori)

    # Allow USD references to finish loading before we proceed.
    simulation_app.update()
    for _ in range(200):
        if not is_stage_loading():
            break
        simulation_app.update()

    return {
        "world": my_world,
        "stage": stage,
        "arm_left": arm_left,
        "arm_right": arm_right,
        "stage_units": stage_units,
    }


def _add_particle_cloth(*, simulation_app, world, cloth_path: str, cloth_center: np.ndarray, dimx: int, dimy: int) -> Any:
    # Use a garment mesh + surface deformable body instead of legacy particle cloth.
    import omni.usd
    from omni.physx.scripts import deformableUtils
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, PhysxSchema, Usd, UsdGeom

    stage = simulation_app.context.get_stage()
    myvla_root = Path(__file__).resolve().parents[1]
    garment_dir = myvla_root / "isaac_sim_assets" / "garments"
    garment_dir.mkdir(parents=True, exist_ok=True)
    garment_obj_path = garment_dir / "shirt_surface.obj"
    garment_usd_path = garment_dir / "Field_Jacket.usd"

    def create_tshirt_shell_mesh(
        *, nx: int, ny: int, height: float, width: float
    ) -> tuple[list[tuple[float, float, float]], list[int]]:
        nx = int(nx)
        ny = int(ny)
        if nx < 4 or ny < 4:
            raise ValueError("cloth_dimx/cloth_dimy must be >= 4 for tshirt mesh")

        xs = np.linspace(-height / 2.0, height / 2.0, nx, dtype=np.float32)
        ys = np.linspace(-width / 2.0, width / 2.0, ny, dtype=np.float32)

        top = float(xs[-1])
        bottom = float(xs[0])
        shoulder_x = top - 0.17 * height
        armhole_x = top - 0.34 * height

        outline_norm = [
            (0.00, -0.27),
            (0.14, -0.27),
            (0.34, -0.29),
            (0.54, -0.32),
            (0.64, -0.41),
            (0.71, -0.60),
            (0.78, -0.82),
            (0.85, -0.98),
            (0.91, -0.87),
            (0.86, -0.61),
            (0.81, -0.48),
            (0.91, -0.38),
            (0.98, -0.24),
            (1.00, -0.16),
            (1.00, 0.16),
            (0.98, 0.24),
            (0.91, 0.38),
            (0.81, 0.48),
            (0.86, 0.61),
            (0.91, 0.87),
            (0.85, 0.98),
            (0.78, 0.82),
            (0.71, 0.60),
            (0.64, 0.41),
            (0.54, 0.32),
            (0.34, 0.29),
            (0.14, 0.27),
            (0.00, 0.27),
        ]
        outline = [(bottom + xn * height, yn * 0.5 * width) for xn, yn in outline_norm]

        def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
            inside_flag = False
            for idx in range(len(polygon)):
                x1, y1 = polygon[idx]
                x2, y2 = polygon[(idx + 1) % len(polygon)]
                if (y1 > y) != (y2 > y):
                    cross_x = x1 + (y - y1) * (x2 - x1) / max(y2 - y1, 1e-6)
                    if x < cross_x:
                        inside_flag = not inside_flag
            return inside_flag

        def neckline_half_width(xn: float) -> float:
            collar_open = 0.06 + 0.06 * max(0.0, min(1.0, (xn - 0.92) / 0.08))
            shoulder_return = 0.010 * math.exp(-((xn - 0.975) / 0.016) ** 2)
            return (collar_open - shoulder_return) * 0.5 * width

        def inside(x: float, y: float) -> bool:
            if not point_in_polygon(x, y, outline):
                return False
            xn = float((x - bottom) / max(top - bottom, 1e-6))
            yn = float(y / max(width * 0.5, 1e-6))
            abs_y = abs(y)
            neck_half_w = neckline_half_width(xn)
            if xn >= 0.93 and abs_y <= neck_half_w:
                return False
            if xn < 0.05:
                hem_half_w = (0.24 + 0.03 * xn / 0.05) * width
                if abs(y) > hem_half_w:
                    return False
            if xn > 0.72 and abs(yn) > 0.60:
                cuff_radius = ((xn - 0.86) / 0.13) ** 2 + ((abs(yn) - 0.89) / 0.16) ** 2
                if cuff_radius > 1.15:
                    return False
            return True

        vidx: dict[tuple[int, int], int] = {}
        base_points_xyz: list[tuple[float, float, float]] = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if inside(float(x), float(y)):
                    vidx[(i, j)] = len(base_points_xyz)
                    xn = float((x - bottom) / max(top - bottom, 1e-6))
                    yn = float(y / max(width * 0.5, 1e-6))
                    wrinkle_z = (
                        0.005 * math.sin(math.pi * xn) * math.cos(2.0 * math.pi * yn)
                        + 0.003 * math.sin(3.0 * math.pi * xn + 0.6) * math.sin(math.pi * yn)
                    )
                    sleeve_lift = 0.003 * max(0.0, xn - 0.62) * abs(yn)
                    torso_sag = -0.003 * math.exp(-((xn - 0.45) / 0.22) ** 2) * (1.0 - min(1.0, abs(yn)))
                    base_points_xyz.append((float(x), float(y), float(wrinkle_z + sleeve_lift + torso_sag)))

        front_indices: list[int] = []
        edge_counts: dict[tuple[int, int], tuple[int, int, int]] = {}
        for i in range(nx - 1):
            for j in range(ny - 1):
                k00 = vidx.get((i, j))
                k10 = vidx.get((i + 1, j))
                k01 = vidx.get((i, j + 1))
                k11 = vidx.get((i + 1, j + 1))
                if k00 is None or k10 is None or k01 is None or k11 is None:
                    continue
                # two triangles per quad (counter-clockwise)
                tris = ((k00, k10, k11), (k00, k11, k01))
                for tri in tris:
                    front_indices += [tri[0], tri[1], tri[2]]
                    tri_edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
                    for ea, eb in tri_edges:
                        key = (min(ea, eb), max(ea, eb))
                        if key in edge_counts:
                            old_a, old_b, count = edge_counts[key]
                            edge_counts[key] = (old_a, old_b, count + 1)
                        else:
                            edge_counts[key] = (ea, eb, 1)
        if len(front_indices) < 3:
            raise RuntimeError("tshirt mesh generation produced no triangles; increase cloth dims")

        adjacency: dict[int, set[int]] = {}
        for idx in range(0, len(front_indices), 3):
            a, b, c = front_indices[idx], front_indices[idx + 1], front_indices[idx + 2]
            adjacency.setdefault(a, set()).update((b, c))
            adjacency.setdefault(b, set()).update((a, c))
            adjacency.setdefault(c, set()).update((a, b))
        keep_vertices: set[int] = set()
        visited_vertices: set[int] = set()
        for start_vertex in adjacency:
            if start_vertex in visited_vertices:
                continue
            component: set[int] = set()
            stack = [start_vertex]
            while stack:
                current_vertex = stack.pop()
                if current_vertex in visited_vertices:
                    continue
                visited_vertices.add(current_vertex)
                component.add(current_vertex)
                stack.extend(adjacency.get(current_vertex, ()))
            if len(component) > len(keep_vertices):
                keep_vertices = component
        if not keep_vertices:
            raise RuntimeError("tshirt mesh generation produced no connected garment surface")
        if len(keep_vertices) != len(base_points_xyz):
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(keep_vertices))}
            base_points_xyz = [base_points_xyz[old_idx] for old_idx in sorted(keep_vertices)]
            filtered_front_indices: list[int] = []
            edge_counts = {}
            for idx in range(0, len(front_indices), 3):
                tri = front_indices[idx : idx + 3]
                if any(vertex not in keep_vertices for vertex in tri):
                    continue
                a, b, c = [remap[vertex] for vertex in tri]
                filtered_front_indices += [a, b, c]
                for ea, eb in ((a, b), (b, c), (c, a)):
                    key = (min(ea, eb), max(ea, eb))
                    if key in edge_counts:
                        old_a, old_b, count = edge_counts[key]
                        edge_counts[key] = (old_a, old_b, count + 1)
                    else:
                        edge_counts[key] = (ea, eb, 1)
            front_indices = filtered_front_indices

        shell_gap = 0.013
        collar_rise = 0.013
        sleeve_roll = 0.010
        hem_roll = 0.0035

        def shape_point(x: float, y: float, *, layer_sign: float) -> tuple[float, float, float]:
            xn = float((x - bottom) / max(top - bottom, 1e-6))
            yn = float(y / max(width * 0.5, 1e-6))
            abs_yn = abs(yn)
            torso_loft = 0.0024 * math.exp(-((xn - 0.46) / 0.24) ** 2) * (1.0 - min(1.0, abs_yn * 1.1))
            sleeve_zone = max(0.0, min(1.0, (xn - 0.62) / 0.20)) * max(0.0, min(1.0, (abs_yn - 0.44) / 0.42))
            sleeve_loft = sleeve_roll * sleeve_zone * (0.7 + 0.3 * max(0.0, (abs_yn - 0.65) / 0.30))
            shoulder_loft = 0.0065 * math.exp(-((xn - 0.82) / 0.09) ** 2) * (0.35 + 0.65 * min(1.0, abs_yn))
            hem_lift = hem_roll * max(0.0, 0.14 - xn) * (0.6 + 0.4 * min(1.0, abs_yn * 1.5))
            collar_band = collar_rise * math.exp(-((xn - 0.95) / 0.045) ** 2) * max(0.0, 1.0 - abs_yn / 0.32)
            collar_flap = 0.0095 * math.exp(-((xn - 0.89) / 0.05) ** 2) * math.exp(-((abs_yn - 0.23) / 0.09) ** 2)
            placket_ridge = 0.0048 * max(0.0, min(1.0, (xn - 0.08) / 0.18)) * math.exp(-(yn / 0.045) ** 2)
            seam_ridge = 0.0038 * math.exp(-((abs_yn - 0.26) / 0.030) ** 2) * (0.2 + 0.8 * min(1.0, xn + 0.05))
            cuff_ridge = 0.0045 * math.exp(-((abs_yn - 0.82) / 0.06) ** 2) * max(0.0, xn - 0.69)
            yoke_ridge = 0.0036 * math.exp(-((xn - 0.79) / 0.035) ** 2) * math.exp(-(yn / 0.44) ** 2)
            underarm_dimple = -0.0034 * math.exp(-((xn - 0.66) / 0.06) ** 2) * math.exp(-((abs_yn - 0.42) / 0.05) ** 2)
            wrinkle = (
                0.0017 * math.sin(2.0 * math.pi * xn + 0.5 * layer_sign) * math.cos(math.pi * yn)
                + 0.0013 * math.sin(4.0 * math.pi * xn + 0.7) * math.sin(1.5 * math.pi * yn)
                + 0.0010 * math.sin(6.0 * math.pi * xn + 1.4 * abs_yn)
            )
            z = (
                layer_sign * 0.5 * shell_gap
                + torso_loft
                + sleeve_loft
                + shoulder_loft
                + hem_lift
                + collar_band
                + collar_flap
                + seam_ridge
                + cuff_ridge
                + yoke_ridge
                + underarm_dimple
                + wrinkle
            )
            if layer_sign > 0.0:
                z += placket_ridge
            if layer_sign < 0.0:
                z -= 0.0020 * math.exp(-((xn - 0.63) / 0.18) ** 2)
                z += 0.0012 * yoke_ridge

            sleeve_push = 0.017 * sleeve_zone
            cuff_pinched = 0.010 * max(0.0, min(1.0, (xn - 0.79) / 0.14)) * math.exp(-((abs_yn - 0.87) / 0.08) ** 2)
            collar_spread = 0.011 * math.exp(-((xn - 0.91) / 0.045) ** 2) * math.exp(-((abs_yn - 0.20) / 0.08) ** 2)
            side_taper = 0.012 * math.exp(-((xn - 0.44) / 0.20) ** 2) * math.exp(-((abs_yn - 0.30) / 0.08) ** 2)
            shoulder_pull = 0.008 * math.exp(-((xn - 0.84) / 0.06) ** 2) * math.exp(-((abs_yn - 0.42) / 0.12) ** 2)
            waist_suppression = 0.009 * math.exp(-((xn - 0.30) / 0.16) ** 2) * math.exp(-((abs_yn - 0.28) / 0.08) ** 2)

            x_adj = x
            y_adj = y + math.copysign(sleeve_push + collar_spread - cuff_pinched - side_taper - waist_suppression, y)
            y_adj += math.copysign(0.010 * shoulder_pull, y)
            x_adj -= 0.032 * sleeve_zone
            x_adj += 0.005 * collar_spread
            x_adj -= 0.007 * shoulder_pull
            if layer_sign > 0.0:
                x_adj += 0.0015 * math.exp(-((xn - 0.86) / 0.08) ** 2) * math.exp(-(yn / 0.18) ** 2)
            else:
                x_adj -= 0.0015 * math.exp(-((xn - 0.80) / 0.07) ** 2) * math.exp(-(yn / 0.28) ** 2)

            return float(x_adj), float(y_adj), float(z)

        front_points = [shape_point(x, y, layer_sign=+1.0) for x, y, _z in base_points_xyz]
        back_points = [shape_point(x, y, layer_sign=-1.0) for x, y, _z in base_points_xyz]
        points = front_points + back_points
        n_base = len(base_points_xyz)
        indices: list[int] = list(front_indices)
        for idx in range(0, len(front_indices), 3):
            a, b, c = front_indices[idx], front_indices[idx + 1], front_indices[idx + 2]
            indices += [c + n_base, b + n_base, a + n_base]

        def is_open_boundary(xm: float, ym: float) -> bool:
            if xm < bottom + 0.08 * height:
                return True
            if xm > top - 0.18 * height and abs(ym) < 0.18 * width:
                return True
            if xm > shoulder_x + 0.01 * height and abs(ym) > 0.34 * width:
                return True
            return False

        for _key, (ea, eb, count) in edge_counts.items():
            if count != 1:
                continue
            ax, ay, _ = base_points_xyz[ea]
            bx, by, _ = base_points_xyz[eb]
            xm = 0.5 * (ax + bx)
            ym = 0.5 * (ay + by)
            if is_open_boundary(float(xm), float(ym)):
                continue
            a_back = ea + n_base
            b_back = eb + n_base
            indices += [ea, eb, b_back, ea, b_back, a_back]

        return points, indices

    def write_obj(path: Path, points_xyz: list[tuple[float, float, float]], indices: list[int]) -> None:
        lines = ["# myVLA procedural shirt surface mesh"]
        for x, y, z in points_xyz:
            lines.append(f"v {float(x):.7f} {float(y):.7f} {float(z):.7f}")
        for idx in range(0, len(indices), 3):
            a, b, c = indices[idx], indices[idx + 1], indices[idx + 2]
            lines.append(f"f {a + 1} {b + 1} {c + 1}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int]]:
        points: list[Gf.Vec3f] = []
        indices: list[int] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, sx, sy, sz = line.split()[:4]
                points.append(Gf.Vec3f(float(sx), float(sy), float(sz)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = [int(part.split("/")[0]) - 1 for part in parts]
                if len(face) == 3:
                    indices.extend(face)
                elif len(face) > 3:
                    for tri_idx in range(1, len(face) - 1):
                        indices.extend([face[0], face[tri_idx], face[tri_idx + 1]])
        if not points or not indices:
            raise RuntimeError(f"Failed to load garment OBJ: {path}")
        return points, indices

    def load_usd_mesh(path: Path, *, target_height: float, target_width: float, target_depth: float = 0.018) -> tuple[list[Gf.Vec3f], list[int]]:
        usd_stage = Usd.Stage.Open(os.fspath(path))
        if usd_stage is None:
            raise RuntimeError(f"Failed to open garment USD: {path}")
        mesh_prim = next((prim for prim in usd_stage.Traverse() if prim.IsA(UsdGeom.Mesh)), None)
        if mesh_prim is None:
            raise RuntimeError(f"No mesh prim found in garment USD: {path}")
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if not points_attr or not counts_attr or not indices_attr:
            raise RuntimeError(f"Garment USD mesh is empty: {path}")

        raw_points = np.asarray([[float(point[0]), float(point[1]), float(point[2])] for point in points_attr], dtype=np.float32)
        mins = raw_points.min(axis=0)
        maxs = raw_points.max(axis=0)
        center = 0.5 * (mins + maxs)
        extents = np.maximum(maxs - mins, 1e-6)

        normalized = raw_points - center
        normalized_x = normalized[:, 2] / extents[2] * float(target_height)
        normalized_y = normalized[:, 0] / extents[0] * float(target_width)
        normalized_z = normalized[:, 1] / extents[1] * float(target_depth)
        remapped_points = np.stack([normalized_x, normalized_y, normalized_z], axis=1)

        tri_indices: list[int] = []
        cursor = 0
        for face_count in counts_attr:
            face = [int(index) for index in indices_attr[cursor : cursor + face_count]]
            cursor += face_count
            if face_count == 3:
                tri_indices.extend(face)
            elif face_count > 3:
                for tri_idx in range(1, face_count - 1):
                    tri_indices.extend([face[0], face[tri_idx], face[tri_idx + 1]])

        if not tri_indices:
            raise RuntimeError(f"Garment USD had no triangulated faces: {path}")
        return [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in remapped_points], tri_indices

    # Heuristic shirt size (meters in stage units). Fits on the table and is easy to see.
    shirt_height = 0.74
    shirt_width = 0.66
    garment_asset_path = garment_obj_path
    if garment_usd_path.exists():
        obj_points, obj_indices = load_usd_mesh(
            garment_usd_path,
            target_height=shirt_height,
            target_width=shirt_width,
            target_depth=0.018,
        )
        garment_asset_path = garment_usd_path
    else:
        points_xyz, tri_indices = create_tshirt_shell_mesh(
            nx=max(int(dimx), 65),
            ny=max(int(dimy), 57),
            height=shirt_height,
            width=shirt_width,
        )
        write_obj(garment_obj_path, points_xyz, tri_indices)
        obj_points, obj_indices = load_obj(garment_obj_path)

    deformable_root = UsdGeom.Xform.Define(stage, cloth_path)
    mesh_path = f"{cloth_path}/mesh"
    sim_mesh_path = f"{cloth_path}/simMesh"
    skin_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    skin_mesh.GetPointsAttr().Set(obj_points)
    skin_mesh.GetFaceVertexIndicesAttr().Set(obj_indices)
    skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(obj_indices) // 3))
    skin_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.80, 0.34, 0.40)])
    skin_mesh.CreateDoubleSidedAttr().Set(True)
    physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)

    root_xform = UsdGeom.Xformable(deformable_root)
    physicsUtils.setup_transform_as_scale_orient_translate(root_xform)
    physicsUtils.set_or_add_translate_op(root_xform, Gf.Vec3f(float(cloth_center[0]), float(cloth_center[1]), float(cloth_center[2])))
    physicsUtils.set_or_add_orient_op(root_xform, Gf.Quatf(1.0))
    physicsUtils.set_or_add_scale_op(root_xform, Gf.Vec3f(1.0))

    scene_prim = stage.GetPrimAtPath(world.get_physics_context().prim_path)
    scene_prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
    scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
    if scene_api.GetTimeStepsPerSecondAttr().Get() is None:
        scene_api.GetTimeStepsPerSecondAttr().Set(120)
    scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)

    ok = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=deformable_root.GetPath(),
        simulation_mesh_path=sim_mesh_path,
        cooking_src_mesh_path=skin_mesh.GetPath(),
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError(f"Failed to create surface deformable garment from {garment_asset_path}")

    deformable_prim = deformable_root.GetPrim()
    deformable_prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    deformable_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    deformable_prim.GetAttribute("physxDeformableBody:selfCollision").Set(True)
    deformable_prim.GetAttribute("physxDeformableBody:selfCollisionFilterDistance").Set(0.012)
    deformable_prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    deformable_prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(20)
    deformable_prim.GetAttribute("physxDeformableBody:linearDamping").Set(0.02)
    deformable_prim.GetAttribute("physxDeformableBody:settlingDamping").Set(0.0)
    deformable_prim.GetAttribute("physxDeformableBody:sleepThreshold").Set(0.0)
    deformable_prim.GetAttribute("physxDeformableBody:settlingThreshold").Set(0.0)
    deformable_prim.GetAttribute("physxDeformableBody:collisionPairUpdateFrequency").Set(1)
    deformable_prim.GetAttribute("physxDeformableBody:collisionIterationMultiplier").Set(2)
    deformable_prim.GetAttribute("physxDeformableBody:maxDepenetrationVelocity").Set(8.0)
    deformable_prim.GetAttribute("physxDeformableBody:maxLinearVelocity").Set(6.0)

    sim_mesh_prim = stage.GetPrimAtPath(sim_mesh_path)
    sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(sim_mesh_prim)
    collision_api.GetRestOffsetAttr().Set(0.002)
    collision_api.GetContactOffsetAttr().Set(0.010)

    material_path = omni.usd.get_stage_next_free_path(stage, "/World/ShirtDeformableMaterial", True)
    deformableUtils.add_surface_deformable_material(
        stage,
        material_path,
        dynamic_friction=0.65,
        youngs_modulus=4000.0,
        poissons_ratio=0.35,
        surface_thickness=0.004,
        surface_stretch_stiffness=180.0,
        surface_shear_stiffness=28.0,
        surface_bend_stiffness=0.08,
    )
    physicsUtils.add_physics_material_to_prim(stage, deformable_prim, material_path)
    material_prim = stage.GetPrimAtPath(material_path)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.02)
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.2)

    return {
        "type": "surface_deformable",
        "root_path": str(deformable_root.GetPath()),
        "mesh_path": mesh_path,
        "sim_mesh_path": sim_mesh_path,
        "asset_path": os.fspath(garment_asset_path),
    }


def _arm_state_from_articulation(arm) -> tuple[np.ndarray, np.ndarray]:
    q = _to_numpy(arm.get_joint_positions())[0]  # [dof]
    if q.shape[0] < 7:
        raise ValueError(f"Expected arm dof>=7, got dof={q.shape[0]}")
    joints7 = q[:7].astype(np.float32)

    # Franka gripper joints vary by URDF/import config:
    # - 9 dof: 7 arm + 2 fingers
    # - 8 dof: 7 arm + 1 gripper
    # - 7 dof: arm-only (no gripper)
    if q.shape[0] >= 9:
        gripper = float(np.mean(q[-2:]))
    elif q.shape[0] == 8:
        gripper = float(q[-1])
    else:
        gripper = 0.0
    return joints7, np.asarray([gripper], dtype=np.float32)


def _apply_action_to_franka(
    arm,
    action_8: np.ndarray,
    *,
    joint_delta_scale: float,
    gripper_delta_scale: float,
    control_dt: float,
) -> None:
    a = np.asarray(action_8, dtype=np.float32).reshape(-1)
    if a.shape[0] < 8:
        raise ValueError(f"Expected action with >=8 dims, got {a.shape}")

    q_raw = arm.get_joint_positions()
    q = _to_numpy(q_raw)[0].astype(np.float32)
    if q.shape[0] < 7:
        raise ValueError(f"Expected arm dof>=7, got dof={q.shape[0]}")
    a = np.clip(a, -1.0, 1.0)

    # pi0.5-DROID outputs 7 joint controls plus 1 gripper command. For the arm, we execute this
    # as joint-velocity targets over the next control interval instead of teleporting joint positions.
    vel = np.zeros_like(q, dtype=np.float32)
    eff_dt = max(float(control_dt), 1e-3)
    vel[:7] = (a[:7] * float(joint_delta_scale)) / eff_dt

    gripper_open = 0.04 if float(a[7]) > 0.5 else 0.0

    if hasattr(q_raw, "to"):
        import torch

        dev = getattr(q_raw, "device", torch.device("cpu"))
        vel_t = torch.as_tensor(vel, dtype=torch.float32, device=dev)[None, :]
        arm.set_joint_velocity_targets(vel_t)
        if q.shape[0] >= 9:
            grip_t = torch.as_tensor([[gripper_open, gripper_open]], dtype=torch.float32, device=dev)
            arm.set_joint_position_targets(grip_t, joint_indices=torch.as_tensor([q.shape[0] - 2, q.shape[0] - 1], dtype=torch.int64, device=dev))
        elif q.shape[0] == 8:
            grip_t = torch.as_tensor([[gripper_open]], dtype=torch.float32, device=dev)
            arm.set_joint_position_targets(grip_t, joint_indices=torch.as_tensor([q.shape[0] - 1], dtype=torch.int64, device=dev))
    else:
        arm.set_joint_velocity_targets([vel.tolist()])
        if q.shape[0] >= 9:
            arm.set_joint_position_targets([[gripper_open, gripper_open]], joint_indices=np.array([q.shape[0] - 2, q.shape[0] - 1], dtype=np.int64))
        elif q.shape[0] == 8:
            arm.set_joint_position_targets([[gripper_open]], joint_indices=np.array([q.shape[0] - 1], dtype=np.int64))


def _set_arm_joint_targets(
    arm,
    *,
    joint_targets_7: np.ndarray,
    gripper_open: float,
) -> None:
    q_raw = arm.get_joint_positions()
    q = _to_numpy(q_raw)[0].astype(np.float32)
    targets = q.copy()
    targets[:7] = np.asarray(joint_targets_7, dtype=np.float32).reshape(7)
    gripper_open = float(np.clip(gripper_open, 0.0, 0.04))
    if q.shape[0] >= 9:
        targets[-2:] = gripper_open
    elif q.shape[0] == 8:
        targets[-1] = gripper_open

    if hasattr(q_raw, "to"):
        import torch

        dev = getattr(q_raw, "device", torch.device("cpu"))
        target_t = torch.as_tensor(targets, dtype=torch.float32, device=dev)[None, :]
        arm.set_joint_position_targets(target_t)
    else:
        arm.set_joint_position_targets([targets.tolist()])


def _solve_franka_ik_targets(
    arm,
    *,
    ee_index: int | None,
    goal_position: np.ndarray,
    goal_orientation: np.ndarray | None = None,
    residual_action: np.ndarray | None = None,
    residual_scale: float = 0.0,
) -> np.ndarray | None:
    if ee_index is None:
        return None
    try:
        q_raw = arm.get_joint_positions()
        q = _to_numpy(q_raw).astype(np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] < 7:
            return None

        idx = np.array([int(ee_index)], dtype=np.int64)
        pos, ori = arm.get_body_coms(body_indices=idx)
        cur_pos = _to_numpy(pos).astype(np.float32)[:, 0, :]
        cur_ori = _to_numpy(ori).astype(np.float32)[:, 0, :]

        jac_all = _to_numpy(arm.get_jacobian_matrices()).astype(np.float32)
        jac_index = max(0, min(int(ee_index) - 1, jac_all.shape[1] - 1))
        jac = jac_all[:, jac_index, :, :7]

        goal_pos = np.asarray(goal_position, dtype=np.float32).reshape(1, 3)
        goal_ori = None if goal_orientation is None else np.asarray(goal_orientation, dtype=np.float32).reshape(1, 4)
        dq = _differential_inverse_kinematics(
            jacobian_end_effector=jac,
            current_position=cur_pos,
            current_orientation=cur_ori,
            goal_position=goal_pos,
            goal_orientation=goal_ori,
        )[0]
        joint_targets = q[0, :7] + np.clip(dq, -0.20, 0.20)
        if residual_action is not None and float(residual_scale) > 0.0:
            residual = np.asarray(residual_action, dtype=np.float32).reshape(-1)
            joint_targets = joint_targets + float(residual_scale) * np.clip(residual[:7], -1.0, 1.0) * 0.04
        return joint_targets.astype(np.float32)
    except Exception:
        return None


def _demo_dual_arm_actions(*, step: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    horizon = max(1, int(horizon))
    t = np.arange(horizon, dtype=np.float32) + float(step) * float(horizon)
    phase = 0.18 * t
    left = np.zeros((horizon, 8), dtype=np.float32)
    right = np.zeros((horizon, 8), dtype=np.float32)
    left[:, 1] = 0.45 * np.sin(phase)
    left[:, 3] = 0.30 * np.cos(phase * 0.8)
    left[:, 5] = -0.25 * np.sin(phase * 0.6)
    right[:, 1] = -0.45 * np.sin(phase)
    right[:, 3] = -0.30 * np.cos(phase * 0.8)
    right[:, 5] = 0.25 * np.sin(phase * 0.6)
    left[:, 7] = 1.0
    right[:, 7] = 1.0
    return left, right


def _fold_targets_for_step(*, step: int, cloth_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    phase_name = fold_phase_name(int(step))
    cloth_center = np.asarray(cloth_center, dtype=np.float32).reshape(3)

    if step < 8:
        left = cloth_center + np.asarray([0.05, 0.18, 0.20], dtype=np.float32)
        right = cloth_center + np.asarray([0.05, -0.18, 0.20], dtype=np.float32)
        gripper_open = 0.04
    elif step < 16:
        left = cloth_center + np.asarray([0.02, 0.15, 0.06], dtype=np.float32)
        right = cloth_center + np.asarray([0.02, -0.15, 0.06], dtype=np.float32)
        gripper_open = 0.04
    elif step < 28:
        inward = 0.15 - 0.008 * float(step - 16)
        left = cloth_center + np.asarray([0.00, inward, 0.015], dtype=np.float32)
        right = cloth_center + np.asarray([0.00, -inward, 0.015], dtype=np.float32)
        gripper_open = 0.0
    elif step < 40:
        inward = max(0.03, 0.07 - 0.004 * float(step - 28))
        lift = 0.05 + 0.010 * float(step - 28)
        forward = -0.015 * float(step - 28)
        left = cloth_center + np.asarray([forward, inward, lift], dtype=np.float32)
        right = cloth_center + np.asarray([forward, -inward, lift], dtype=np.float32)
        gripper_open = 0.0
    elif step < 50:
        left = cloth_center + np.asarray([-0.09, 0.03, 0.020], dtype=np.float32)
        right = cloth_center + np.asarray([-0.09, -0.03, 0.020], dtype=np.float32)
        gripper_open = 0.02
    else:
        left = cloth_center + np.asarray([0.02, 0.13, 0.14], dtype=np.float32)
        right = cloth_center + np.asarray([0.02, -0.13, 0.14], dtype=np.float32)
        gripper_open = 0.04

    return left.astype(np.float32), right.astype(np.float32), float(gripper_open), phase_name


def _scripted_fold_actions(*, step: int, horizon: int) -> tuple[np.ndarray, np.ndarray, str]:
    horizon = max(1, int(horizon))
    phase_name = fold_phase_name(int(step))
    left = np.zeros((horizon, 8), dtype=np.float32)
    right = np.zeros((horizon, 8), dtype=np.float32)

    if step < 8:
        left_base = np.asarray([0.25, 0.55, 0.00, 0.85, 0.00, -0.45, 0.00, 1.0], dtype=np.float32)
        right_base = np.asarray([-0.25, 0.55, 0.00, 0.85, 0.00, 0.45, 0.00, 1.0], dtype=np.float32)
    elif step < 16:
        left_base = np.asarray([0.10, 0.15, 0.00, 0.95, 0.00, -0.55, 0.00, 0.0], dtype=np.float32)
        right_base = np.asarray([-0.10, 0.15, 0.00, 0.95, 0.00, 0.55, 0.00, 0.0], dtype=np.float32)
    elif step < 26:
        left_base = np.asarray([-0.65, 0.10, 0.00, -0.15, 0.00, 0.70, 0.00, 0.0], dtype=np.float32)
        right_base = np.asarray([0.65, 0.10, 0.00, -0.15, 0.00, -0.70, 0.00, 0.0], dtype=np.float32)
    elif step < 38:
        left_base = np.asarray([0.35, -0.25, 0.00, -0.70, 0.00, 0.35, 0.00, 0.0], dtype=np.float32)
        right_base = np.asarray([-0.35, -0.25, 0.00, -0.70, 0.00, -0.35, 0.00, 0.0], dtype=np.float32)
    elif step < 50:
        left_base = np.asarray([0.15, -0.10, 0.00, 0.30, 0.00, -0.20, 0.00, 1.0], dtype=np.float32)
        right_base = np.asarray([-0.15, -0.10, 0.00, 0.30, 0.00, 0.20, 0.00, 1.0], dtype=np.float32)
    else:
        left_base = np.asarray([-0.20, -0.20, 0.00, 0.45, 0.00, -0.10, 0.00, 1.0], dtype=np.float32)
        right_base = np.asarray([0.20, -0.20, 0.00, 0.45, 0.00, 0.10, 0.00, 1.0], dtype=np.float32)

    for idx in range(horizon):
        wobble = 0.10 * math.sin(0.35 * (float(step) * float(horizon) + float(idx)))
        left[idx] = left_base
        right[idx] = right_base
        left[idx, 2] = wobble
        right[idx, 2] = -wobble
        left[idx, 6] = 0.12 * math.cos(0.21 * (float(step) * float(horizon) + float(idx)))
        right[idx, 6] = -left[idx, 6]

    return np.clip(left, -1.0, 1.0), np.clip(right, -1.0, 1.0), phase_name


def _update_shirt_fold_visual(*, stage: Any, cloth_root_path: str, step: int, cloth_center: np.ndarray) -> None:
    try:
        from pxr import Gf, UsdGeom

        prim = stage.GetPrimAtPath(str(cloth_root_path))
        if not prim or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        translate_op = None
        scale_op = None
        rotate_op = None
        for op in xf.GetOrderedXformOps():
            op_type = op.GetOpType()
            if op_type == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op_type == UsdGeom.XformOp.TypeScale:
                scale_op = op
            elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        if scale_op is None:
            scale_op = xf.AddScaleOp()
        if rotate_op is None:
            rotate_op = xf.AddRotateXYZOp()

        progress = float(np.clip((int(step) - 16) / 34.0, 0.0, 1.0))
        center = np.asarray(cloth_center, dtype=np.float32).reshape(3)
        translate = center.copy()
        translate[0] += -0.05 * progress
        translate[2] += 0.018 * math.sin(progress * math.pi)
        scale = np.asarray(
            [
                1.0 - 0.10 * progress,
                1.0 - 0.52 * progress,
                1.0,
            ],
            dtype=np.float32,
        )
        rotate = np.asarray([0.0, 0.0, 4.0 * progress], dtype=np.float32)

        translate_op.Set(Gf.Vec3f(float(translate[0]), float(translate[1]), float(translate[2])))
        scale_op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
        rotate_op.Set(Gf.Vec3f(float(rotate[0]), float(rotate[1]), float(rotate[2])))
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim headless dual-arm + cloth + myVLA(pi0.5+memory) demo.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (recommended).")
    parser.add_argument("--renderer", default="RayTracedLighting", help="Renderer (default: RayTracedLighting)")
    parser.add_argument("--active_gpu", type=int, default=-1, help="Optional Kit renderer active GPU index.")
    parser.add_argument("--physics_gpu", type=int, default=-1, help="Optional PhysX CUDA GPU index.")
    parser.add_argument(
        "--disable_fabric_delegate",
        action="store_true",
        help="Pass --/app/useFabricSceneDelegate=0 (sometimes needed for cloth).",
    )

    parser.add_argument("--no_myvla", action="store_true", help="Skip myVLA imports/inference (debug).")
    parser.add_argument("--no_cloth", action="store_true", help="Skip particle cloth (debug).")
    default_policy_mode = "rpc" if sys.platform.startswith("win") else "inproc"
    parser.add_argument(
        "--policy_mode",
        default=default_policy_mode,
        choices=("rpc", "inproc", "none"),
        help="Policy integration mode. On Windows, default uses RPC to avoid sentencepiece/Kit crashes.",
    )
    parser.add_argument("--rpc_host", default="127.0.0.1", help="RPC host for policy_mode=rpc")
    parser.add_argument("--rpc_port", type=int, default=5555, help="RPC port for policy_mode=rpc")
    parser.add_argument("--rpc_timeout_s", type=float, default=300.0, help="Socket timeout for policy_mode=rpc")
    parser.add_argument("--rpc_auto_start", action="store_true", help="Spawn the policy RPC server automatically.")
    parser.add_argument("--rpc_start_timeout_s", type=float, default=600.0, help="Seconds to wait for auto-started RPC server.")
    parser.add_argument("--rpc_python_exe", default="python", help="Python executable used for auto-started RPC server.")
    parser.add_argument("--rpc_cuda_visible_devices", default="", help="Optional CUDA_VISIBLE_DEVICES for the RPC server.")
    parser.add_argument("--rpc_policy_device", default="", help="Torch device for RPC pi0.5 policy (e.g. cuda:0).")
    parser.add_argument("--rpc_hl_device", default="", help="Torch device for RPC high-level VLM (e.g. cuda:1).")
    parser.add_argument("--rpc_log_path", default="", help="Optional log file for the auto-started RPC server.")
    parser.add_argument(
        "--rpc_state_file",
        default=str(Path(__file__).resolve().parents[1] / "isaac_sim_runtime" / "rpc_server_state.json"),
        help="Metadata file used to discover/reuse a warm RPC server.",
    )
    parser.add_argument("--rpc_no_reuse_existing", action="store_true", help="Do not reuse an already-running warm RPC server.")
    parser.add_argument("--rpc_close_on_exit", action="store_true", help="Send cmd=close and stop the RPC server on exit.")

    parser.add_argument("--checkpoint_dir", default=str(Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"))
    parser.add_argument("--tokenizer_model", default="", help="Optional paligemma_tokenizer.model path")
    parser.add_argument("--device", default="", help="Torch device for pi0.5 (e.g. cuda:0/cpu). Default: auto")
    parser.add_argument("--num_steps", type=int, default=4, help="Flow matching steps per low-level call (default: 4)")
    parser.add_argument("--video_window", type=int, default=4, help="Short-term video memory window")
    parser.add_argument(
        "--task_adapter_residual_scale",
        type=float,
        default=0.20,
        help="How much raw pi0.5 action residual to mix into the scripted fold controller.",
    )
    parser.add_argument(
        "--mem_steps",
        type=int,
        default=25,
        help="How many outer-loop interaction steps (set <=0 to run until closed / Ctrl+C).",
    )
    parser.add_argument(
        "--print_cam_every",
        type=int,
        default=1,
        help="Print camera positions every N outer steps (0=never).",
    )
    parser.add_argument("--camera_warmup_steps", type=int, default=5, help="Render steps before reading cameras.")
    parser.add_argument("--camera_read_attempts", type=int, default=8, help="Max retries when reading camera RGB.")
    parser.add_argument("--overview_focal_length", type=float, default=3.0, help="Optional overview camera focal length.")
    parser.add_argument(
        "--exterior_focal_length",
        type=float,
        default=3.0,
        help="Optional shared focal length for left/right exterior cameras.",
    )
    parser.add_argument(
        "--wrist_focal_length",
        type=float,
        default=1.8,
        help="Optional shared focal length for left/right wrist cameras.",
    )

    # Long-term memory (HL VLM)
    parser.add_argument("--hl_vlm_dir", default="", help="Pretrained VLM directory/id (optional)")
    parser.add_argument("--hl_device", default="cpu", help="HL VLM device (default: cpu)")
    parser.add_argument("--hl_dtype", default="bfloat16", help="HL VLM dtype: bfloat16/float32")
    parser.add_argument("--hl_revision", default="", help="HF revision for HL VLM (optional)")
    parser.add_argument("--hl_max_new_tokens", type=int, default=64)
    parser.add_argument("--hl_temperature", type=float, default=0.0)

    parser.add_argument("--goal", default="fold the shirt", help="Task instruction (natural language)")

    # Sim settings
    parser.add_argument("--backend", default="torch", help="World backend: torch/numpy (default: torch)")
    parser.add_argument("--sim_device", default="cuda", help="World device (default: cuda)")
    parser.add_argument("--gpu_rank", type=int, default=0, help="Local rank hint for multi-GPU scheduling.")
    parser.add_argument("--world_size", type=int, default=1, help="World-size hint for future multi-GPU rollout.")
    parser.add_argument("--sim_steps_per_action", type=int, default=8, help="Physics steps per action in the chunk")
    parser.add_argument(
        "--execute_horizon",
        type=int,
        default=8,
        help="How many actions from each chunk to execute before re-planning (<=0: execute full chunk).",
    )
    parser.add_argument("--joint_delta_scale", type=float, default=0.05, help="Scale for first 7 action dims")
    parser.add_argument("--gripper_delta_scale", type=float, default=0.005, help="Scale for gripper action dim")
    parser.add_argument("--table_z", type=float, default=0.5, help="Table center z position (meters)")
    parser.add_argument("--cloth_z", type=float, default=0.8, help="Cloth center z position (meters)")
    parser.add_argument("--cloth_dimx", type=int, default=31)
    parser.add_argument("--cloth_dimy", type=int, default=29)
    parser.add_argument("--cloth_settle_steps", type=int, default=18, help="Physics steps to let the shirt relax before inference.")
    parser.add_argument(
        "--stay_open",
        action="store_true",
        help="When running with GUI (no --headless), keep stepping the sim until the user closes the window.",
    )

    # Viz dump
    parser.add_argument("--viz_dir", default="", help="Output directory (default: myVLA/isaac_sim_viz/<timestamp>)")
    parser.add_argument("--viz_name", default="", help="Run folder name (optional)")
    args, _unknown = parser.parse_known_args()
    if int(args.rpc_port) == 0:
        args.rpc_port = _pick_free_port(str(args.rpc_host))

    launch_config: dict[str, Any] = {"headless": bool(args.headless), "renderer": str(args.renderer)}
    extra_args: list[str] = []
    if bool(args.disable_fabric_delegate):
        extra_args.append("--/app/useFabricSceneDelegate=0")
    if int(getattr(args, "active_gpu", -1)) >= 0:
        extra_args.append(f"--/renderer/activeGpu={int(args.active_gpu)}")
    if int(getattr(args, "physics_gpu", -1)) >= 0:
        extra_args.append(f"--/physics/cudaDevice={int(args.physics_gpu)}")
    if extra_args:
        launch_config["extra_args"] = extra_args

    myvla_root = _add_myvla_to_syspath()

    policy_mode = str(getattr(args, "policy_mode", "inproc")).lower()
    if bool(args.no_myvla):
        policy_mode = "none"
    args.policy_mode = policy_mode

    PretrainedVlmLongTermMemoryProcessor = None
    InferenceVizWriter = None
    Pi05DroidPolicy = None
    if policy_mode != "rpc":
        # Import myVLA *before* instantiating SimulationApp.
        #
        # On some setups, importing `sentencepiece` after Omniverse/Kit is fully initialized can hard-crash
        # the process due to native dependency conflicts. Importing myVLA (and thus sentencepiece) first is
        # a pragmatic workaround.
        try:
            import sentencepiece  # noqa: F401
        except Exception:  # noqa: BLE001
            pass

        try:
            from myvla_mem.viz import InferenceVizWriter as _InferenceVizWriter
            _PretrainedVlmLongTermMemoryProcessor = None
            _Pi05DroidPolicy = None
            if policy_mode == "inproc":
                from myvla_mem.long_term import (
                    PretrainedVlmLongTermMemoryProcessor as _PretrainedVlmLongTermMemoryProcessor,
                )
                from myvla_pi05.policy import Pi05DroidPolicy as _Pi05DroidPolicy
        except Exception:  # noqa: BLE001
            _eprint("Failed to import myVLA modules inside Isaac Sim Python.")
            _eprint("You likely need to install myVLA deps into Isaac Sim's Python environment:")
            _eprint("  cd issac-sim")
            _eprint("  .\\python.bat -m pip install -r ..\\myVLA\\requirements.txt")
            _eprint("")
            raise

        PretrainedVlmLongTermMemoryProcessor = _PretrainedVlmLongTermMemoryProcessor
        InferenceVizWriter = _InferenceVizWriter
        Pi05DroidPolicy = _Pi05DroidPolicy

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config)
    try:
        scene = _setup_scene(
            simulation_app=simulation_app,
            backend=str(args.backend),
            device=str(args.sim_device),
            table_z=float(args.table_z),
        )
        world = scene["world"]
        stage = scene["stage"]
        arm_left = scene["arm_left"]
        arm_right = scene["arm_right"]
        stage_units = float(scene.get("stage_units", 1.0) or 1.0)

        # Particle cloth "shirt"
        cloth_root_path = "/World/ShirtCloth"
        if not bool(args.no_cloth):
            _add_particle_cloth(
                simulation_app=simulation_app,
                world=world,
                cloth_path=cloth_root_path,
                cloth_center=np.asarray([0.6, 0.0, float(args.cloth_z)], dtype=np.float32),
                dimx=int(args.cloth_dimx),
                dimy=int(args.cloth_dimy),
            )

        # Cameras
        from isaacsim.sensors.camera import Camera

        res = (224, 224)
        cameras = {
            "overview": Camera(
                prim_path="/World/Cameras/overview",
                position=np.array([1.95, 0.0, 1.95], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                frequency=20,
                resolution=res,
                annotator_device="cpu",
            ),
            "left_exterior": Camera(
                prim_path="/World/Cameras/left_exterior",
                position=np.array([1.30, 1.05, 1.40], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                frequency=20,
                resolution=res,
                annotator_device="cpu",
            ),
            "right_exterior": Camera(
                prim_path="/World/Cameras/right_exterior",
                position=np.array([1.30, -1.05, 1.40], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                frequency=20,
                resolution=res,
                annotator_device="cpu",
            ),
            "left_wrist": Camera(
                prim_path="/World/Cameras/left_wrist",
                position=np.array([0.65, 0.30, 1.0], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                frequency=20,
                resolution=res,
                annotator_device="cpu",
            ),
            "right_wrist": Camera(
                prim_path="/World/Cameras/right_wrist",
                position=np.array([0.65, -0.30, 1.0], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                frequency=20,
                resolution=res,
                annotator_device="cpu",
            ),
        }
        for cam in cameras.values():
            cam.initialize()
        simulation_app.update()
        for cam in cameras.values():
            cam.initialize()
        requested_focals = {
            "overview": float(getattr(args, "overview_focal_length", 0.0)),
            "left_exterior": float(getattr(args, "exterior_focal_length", 0.0)),
            "right_exterior": float(getattr(args, "exterior_focal_length", 0.0)),
            "left_wrist": float(getattr(args, "wrist_focal_length", 0.0)),
            "right_wrist": float(getattr(args, "wrist_focal_length", 0.0)),
        }
        for name, cam in cameras.items():
            if requested_focals[name] > 0.0:
                cam.set_focal_length(float(requested_focals[name]))
        for cam in cameras.values():
            cam.set_clipping_range(0.02, 1.0e5)

        # Initialize physics & articulations
        world.reset(soft=False)
        arm_left.initialize()
        arm_right.initialize()

        # Place arms on opposite sides of the table (do this after reset+initialize; otherwise
        # the placement can be overwritten by reset/default states).
        left_base_pos = np.asarray([[0.05, 0.72, 0.0]], dtype=np.float32)
        right_base_pos = np.asarray([[0.05, -0.72, 0.0]], dtype=np.float32)
        left_base_ori = _yaw_quat_wxyz(-90.0)[None, :]
        right_base_ori = _yaw_quat_wxyz(+90.0)[None, :]
        if str(args.backend).lower() == "torch":
            import torch

            dev = torch.device(str(args.sim_device))
            left_base_pos_t = torch.as_tensor(left_base_pos, dtype=torch.float32, device=dev)
            right_base_pos_t = torch.as_tensor(right_base_pos, dtype=torch.float32, device=dev)
            left_base_ori_t = torch.as_tensor(left_base_ori, dtype=torch.float32, device=dev)
            right_base_ori_t = torch.as_tensor(right_base_ori, dtype=torch.float32, device=dev)
            arm_left.set_world_poses(positions=left_base_pos_t, orientations=left_base_ori_t)
            arm_right.set_world_poses(positions=right_base_pos_t, orientations=right_base_ori_t)
            arm_left.set_default_state(positions=left_base_pos_t, orientations=left_base_ori_t)
            arm_right.set_default_state(positions=right_base_pos_t, orientations=right_base_ori_t)
        else:
            arm_left.set_world_poses(positions=left_base_pos, orientations=left_base_ori)
            arm_right.set_world_poses(positions=right_base_pos, orientations=right_base_ori)
            arm_left.set_default_state(positions=left_base_pos, orientations=left_base_ori)
            arm_right.set_default_state(positions=right_base_pos, orientations=right_base_ori)

        # Also move the container prims (more reliable than articulation root discovery on some imports).
        try:
            from pxr import Gf, UsdGeom

            def _set_container_pose(path: str, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
                prim = stage.GetPrimAtPath(str(path))
                if not prim or not prim.IsValid():
                    return
                xf = UsdGeom.Xformable(prim)
                translate_op = None
                orient_op = None
                for op in xf.GetOrderedXformOps():
                    t = op.GetOpType()
                    if t == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                    elif t == UsdGeom.XformOp.TypeOrient:
                        orient_op = op
                if translate_op is None:
                    translate_op = xf.AddTranslateOp()
                if orient_op is None:
                    orient_op = xf.AddOrientOp()
                p = np.asarray(pos_xyz, dtype=np.float32).reshape(3)
                q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
                translate_op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
                orient_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))

            _set_container_pose("/World/ArmLeft", left_base_pos[0], left_base_ori[0])
            _set_container_pose("/World/ArmRight", right_base_pos[0], right_base_ori[0])
        except Exception:
            pass

        left_home_joints = np.asarray([0.18, -0.82, -0.04, -2.10, 0.06, 1.95, 0.92], dtype=np.float32)
        right_home_joints = np.asarray([-0.18, -0.82, 0.04, -2.10, -0.06, 1.95, -0.92], dtype=np.float32)
        try:
            _set_arm_joint_targets(arm_left, joint_targets_7=left_home_joints, gripper_open=0.04)
            _set_arm_joint_targets(arm_right, joint_targets_7=right_home_joints, gripper_open=0.04)
            for _ in range(16):
                world.step(render=True)
        except Exception:
            pass

        # Helpful debug: confirm the two arms are not overlapping after placement.
        try:
            pL, _qL = arm_left.get_world_poses()
            pR, _qR = arm_right.get_world_poses()
            pL_np = _to_numpy(pL)[0].astype(np.float32)
            pR_np = _to_numpy(pR)[0].astype(np.float32)
            print(f"[arm] left_base_pos={pL_np.tolist()} right_base_pos={pR_np.tolist()} stage_units={stage_units}")
        except Exception:
            pass

        settle_steps = max(0, int(getattr(args, "cloth_settle_steps", 0)))
        if settle_steps > 0 and not bool(args.no_cloth):
            print(f"[cloth] settling shirt for {settle_steps} physics steps")
            for _ in range(settle_steps):
                world.step(render=True)

        # Determine EE body indices for wrist-camera tracking (best-effort).
        def pick_ee_index(arm) -> int | None:
            try:
                names = list(getattr(arm, "body_names", []) or [])
                if not names:
                    return None
                for n in names:
                    if "hand" in str(n).lower():
                        return int(arm.get_body_index(n))
                return int(arm.get_body_index(names[-1]))
            except Exception:
                return None

        ee_left = pick_ee_index(arm_left)
        ee_right = pick_ee_index(arm_right)

        goal = str(args.goal).strip()
        policy = None
        hl = None
        writer = None
        rpc = None
        rpc_server_proc: subprocess.Popen[str] | None = None

        if str(args.policy_mode).lower() != "rpc":
            viz_base = (
                Path(str(args.viz_dir)).expanduser().resolve()
                if str(args.viz_dir).strip()
                else (myvla_root / "isaac_sim_viz")
            )
            meta = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "args": vars(args)}
            writer = InferenceVizWriter.create(base_dir=viz_base, name=str(args.viz_name).strip() or None, meta=meta)
            print(f"[viz] writing to: {writer.run_dir}")
        if str(args.policy_mode).lower() == "inproc":
            # myVLA policy (single shared model; called twice per outer step)
            policy = Pi05DroidPolicy(
                args.checkpoint_dir,
                device=str(args.device).strip() or None,
                tokenizer_model=str(args.tokenizer_model).strip() or None,
            )

            if str(args.hl_vlm_dir).strip():
                hl = PretrainedVlmLongTermMemoryProcessor(
                    str(args.hl_vlm_dir).strip(),
                    device=str(args.hl_device).strip() or None,
                    dtype=str(args.hl_dtype),
                    revision=str(args.hl_revision).strip() or None,
                    max_new_tokens=int(args.hl_max_new_tokens),
                    temperature=float(args.hl_temperature),
                )

        elif str(args.policy_mode).lower() == "rpc":
            from rpc_pickle import RpcClient

            reused_existing_rpc = False
            stale_rpc_pid = 0
            if bool(args.rpc_auto_start):
                rpc_state = None if bool(args.rpc_no_reuse_existing) else _load_rpc_state(str(args.rpc_state_file).strip())
                if rpc_state and rpc_state.get("ready"):
                    state_host = str(rpc_state.get("host", args.rpc_host))
                    state_port = int(rpc_state.get("port", 0) or 0)
                    state_ckpt = str(rpc_state.get("checkpoint_dir", ""))
                    state_window = int(rpc_state.get("video_window", -1) or -1)
                    desired_policy_device = str(args.rpc_policy_device).strip() or str(args.device).strip()
                    desired_hl_device = str(args.rpc_hl_device).strip() or str(args.hl_device).strip()
                    state_policy_device = str(rpc_state.get("device", ""))
                    state_hl_device = str(rpc_state.get("hl_device", ""))
                    state_hl_vlm_dir = str(rpc_state.get("hl_vlm_dir", ""))
                    stale_rpc_pid = int(rpc_state.get("pid", 0) or 0)
                    desired_hl_vlm_dir = str(args.hl_vlm_dir).strip()
                    if (
                        state_port > 0
                        and state_ckpt == str(args.checkpoint_dir)
                        and state_window == int(args.video_window)
                        and state_policy_device == desired_policy_device
                        and state_hl_device == desired_hl_device
                        and state_hl_vlm_dir == desired_hl_vlm_dir
                    ):
                        pong = _rpc_ping(state_host, state_port, min(10.0, float(args.rpc_timeout_s)))
                        if pong and pong.get("ready"):
                            args.rpc_host = state_host
                            args.rpc_port = state_port
                            reused_existing_rpc = True
                            print(f"[rpc] reusing warm server at {args.rpc_host}:{int(args.rpc_port)}")
                    else:
                        print(
                            "[rpc] warm server config mismatch; spawning a new server "
                            f"(state_device={state_policy_device or 'auto'} desired_device={desired_policy_device or 'auto'} "
                            f"state_hl_device={state_hl_device or 'auto'} desired_hl_device={desired_hl_device or 'auto'})"
                        )

                if not reused_existing_rpc:
                    if stale_rpc_pid > 0:
                        try:
                            os.kill(stale_rpc_pid, 0)
                        except OSError:
                            stale_rpc_pid = 0
                        if stale_rpc_pid > 0:
                            print(f"[rpc] stopping stale warm server pid={stale_rpc_pid}")
                            os.kill(stale_rpc_pid, signal.SIGTERM)
                            deadline = time.time() + 10.0
                            while time.time() < deadline:
                                try:
                                    os.kill(stale_rpc_pid, 0)
                                except OSError:
                                    stale_rpc_pid = 0
                                    break
                                time.sleep(0.25)
                            if stale_rpc_pid > 0:
                                raise RuntimeError(f"Failed to stop stale warm RPC server pid={stale_rpc_pid}")
                    server_py = myvla_root / "isaac_sim" / "policy_rpc_server.py"
                    rpc_policy_device = str(args.rpc_policy_device).strip() or str(args.device).strip()
                    rpc_hl_device = str(args.rpc_hl_device).strip() or str(args.hl_device).strip()
                    server_cmd = [
                        str(args.rpc_python_exe),
                        "-u",
                        os.fspath(server_py),
                        "--host",
                        str(args.rpc_host),
                        "--port",
                        str(int(args.rpc_port)),
                        "--checkpoint_dir",
                        str(args.checkpoint_dir),
                        "--video_window",
                        str(int(args.video_window)),
                        "--state_file",
                        str(args.rpc_state_file),
                    ]
                    if str(args.tokenizer_model).strip():
                        server_cmd += ["--tokenizer_model", str(args.tokenizer_model)]
                    if rpc_policy_device:
                        server_cmd += ["--device", rpc_policy_device]
                    if str(args.hl_vlm_dir).strip():
                        server_cmd += ["--hl_vlm_dir", str(args.hl_vlm_dir)]
                        if rpc_hl_device:
                            server_cmd += ["--hl_device", rpc_hl_device]
                        if str(args.hl_dtype).strip():
                            server_cmd += ["--hl_dtype", str(args.hl_dtype)]
                        if str(args.hl_revision).strip():
                            server_cmd += ["--hl_revision", str(args.hl_revision)]
                        server_cmd += ["--hl_max_new_tokens", str(int(args.hl_max_new_tokens))]
                        server_cmd += ["--hl_temperature", str(float(args.hl_temperature))]
                    if str(args.viz_dir).strip():
                        server_cmd += ["--viz_dir", str(args.viz_dir)]
                    if str(args.viz_name).strip():
                        server_cmd += ["--viz_name", f"{str(args.viz_name).strip()}_rpc"]

                    env = os.environ.copy()
                    if str(args.rpc_cuda_visible_devices).strip():
                        env["CUDA_VISIBLE_DEVICES"] = str(args.rpc_cuda_visible_devices).strip()
                    log_path = str(args.rpc_log_path).strip()
                    stdout_handle = None
                    if log_path:
                        Path(log_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                        stdout_handle = open(Path(log_path).expanduser().resolve(), "w", encoding="utf-8")
                    if os.name == "nt" and not bool(args.rpc_close_on_exit):
                        _spawn_detached_windows(
                            server_cmd,
                            cwd=myvla_root,
                            log_path=os.fspath(Path(log_path).expanduser().resolve()) if log_path else None,
                        )
                        rpc_server_proc = None
                    else:
                        popen_kwargs: dict[str, Any] = {
                            "cwd": os.fspath(myvla_root),
                            "env": env,
                            "stdout": stdout_handle if stdout_handle is not None else None,
                            "stderr": subprocess.STDOUT if stdout_handle is not None else None,
                        }
                        rpc_server_proc = subprocess.Popen(server_cmd, **popen_kwargs)  # noqa: S603
                    pong = _wait_for_rpc_ready(str(args.rpc_host), int(args.rpc_port), float(args.rpc_start_timeout_s))
                    print(
                        f"[rpc] auto-started warm server pid={(rpc_server_proc.pid if rpc_server_proc is not None else 'detached')} "
                        f"policy_device={rpc_policy_device or 'auto'} hl_device={rpc_hl_device or 'auto'} "
                        f"ready_port={pong.get('port', int(args.rpc_port))}"
                    )

            rpc = RpcClient(
                host=str(args.rpc_host),
                port=int(args.rpc_port),
                timeout_s=float(args.rpc_timeout_s),
            )
            try:
                resp = rpc.call({"cmd": "reset", "goal": goal, "language_memory": ""})
            except Exception:  # noqa: BLE001
                _eprint(f"[rpc] failed to connect to {args.rpc_host}:{int(args.rpc_port)}")
                _eprint("Start the policy server (system python) in another terminal, e.g.:")
                _eprint("  python myVLA/isaac_sim/policy_rpc_server.py --port 5555")
                raise
            viz_run_dir = resp.get("viz_run_dir")
            if viz_run_dir:
                print(f"[rpc] connected; server viz: {viz_run_dir}")

        stack_left = FrameStacker(window=int(args.video_window))
        stack_right = FrameStacker(window=int(args.video_window))

        language_memory = ""
        subtask = goal
        last_actions_both: np.ndarray | None = None
        cloth_center = np.asarray([0.6, 0.0, float(args.cloth_z) - 0.12], dtype=np.float32)
        cloth_left_focus = cloth_center + np.asarray([0.02, 0.14, 0.04], dtype=np.float32)
        cloth_right_focus = cloth_center + np.asarray([0.02, -0.14, 0.04], dtype=np.float32)
        overview_target = cloth_center + np.asarray([0.00, 0.0, 0.02], dtype=np.float32)
        left_ext_target = cloth_center + np.asarray([0.03, 0.05, 0.02], dtype=np.float32)
        right_ext_target = cloth_center + np.asarray([0.03, -0.05, 0.02], dtype=np.float32)
        control_dt = float(world.get_physics_dt()) * float(max(1, int(args.sim_steps_per_action)))

        max_steps = int(args.mem_steps)
        step = 0
        try:
            while True:
                if max_steps > 0 and step >= max_steps:
                    break
                if not simulation_app.is_running():
                    break

                # Update cameras. We keep:
                # - one global overview camera for high-level memory/debugging
                # - one DROID-like exterior camera per arm
                # - one wrist camera per arm
                overview_pos = np.asarray([0.60, 0.0, 3.10], dtype=np.float32)
                left_ext_pos = np.asarray([0.46, 0.40, 2.90], dtype=np.float32)
                right_ext_pos = np.asarray([0.74, -0.40, 2.90], dtype=np.float32)
                overview_quat = _as_float32_quat(
                    np.asarray([0.70710677, 0.0, 0.70710677, 0.0], dtype=np.float32)
                )
                left_ext_quat = overview_quat
                right_ext_quat = overview_quat
                cameras["overview"].set_world_pose(
                    position=overview_pos,
                    orientation=overview_quat,
                )
                cameras["left_exterior"].set_world_pose(
                    position=left_ext_pos,
                    orientation=left_ext_quat,
                )
                cameras["right_exterior"].set_world_pose(
                    position=right_ext_pos,
                    orientation=right_ext_quat,
                )
                if int(getattr(args, "print_cam_every", 0)) > 0 and (step % int(args.print_cam_every) == 0):
                    print(
                        f"[cam] overview_pos={overview_pos.tolist()} "
                        f"target_hint={overview_target.tolist()}"
                    )
                    print(
                        f"[cam] left_exterior_pos={left_ext_pos.tolist()} "
                        f"target_hint={left_ext_target.tolist()}"
                    )
                    print(
                        f"[cam] right_exterior_pos={right_ext_pos.tolist()} "
                        f"target_hint={right_ext_target.tolist()}"
                    )

                if ee_left is not None:
                    if str(args.backend).lower() == "torch":
                        import torch

                        idx = torch.as_tensor([int(ee_left)], dtype=torch.long, device=torch.device(str(args.sim_device)))
                    else:
                        idx = np.array([int(ee_left)], dtype=np.int64)
                    pos, ori = arm_left.get_body_coms(body_indices=idx)
                    ee_pos = _to_numpy(pos)[0, 0].astype(np.float32)
                    ee_ori = _as_float32_quat(_to_numpy(ori)[0, 0])
                    cam_local = np.asarray([0.14, 0.17, 0.10], dtype=np.float32)
                    cam_pos = ee_pos + _quat_rotate_vec_wxyz(ee_ori, cam_local) + np.asarray([0.03, 0.00, 0.10], dtype=np.float32)
                    wrist_forward = _quat_rotate_vec_wxyz(ee_ori, np.asarray([0.30, 0.0, -0.05], dtype=np.float32))
                    wrist_target = 0.35 * (ee_pos + wrist_forward) + 0.65 * cloth_left_focus
                    cameras["left_wrist"].set_world_pose(
                        position=cam_pos,
                        orientation=_look_at_quat_wxyz(camera_pos=cam_pos, target_pos=wrist_target),
                    )
                    if int(getattr(args, "print_cam_every", 0)) > 0 and (step % int(args.print_cam_every) == 0):
                        print(
                            f"[cam] left_wrist_pos={cam_pos.tolist()} ee_pos={ee_pos.tolist()} "
                            f"look_at={wrist_target.tolist()}"
                        )

                if ee_right is not None:
                    if str(args.backend).lower() == "torch":
                        import torch

                        idx = torch.as_tensor([int(ee_right)], dtype=torch.long, device=torch.device(str(args.sim_device)))
                    else:
                        idx = np.array([int(ee_right)], dtype=np.int64)
                    pos, ori = arm_right.get_body_coms(body_indices=idx)
                    ee_pos = _to_numpy(pos)[0, 0].astype(np.float32)
                    ee_ori = _as_float32_quat(_to_numpy(ori)[0, 0])
                    cam_local = np.asarray([0.14, 0.17, 0.10], dtype=np.float32)
                    cam_pos = ee_pos + _quat_rotate_vec_wxyz(ee_ori, cam_local) + np.asarray([0.03, -0.04, 0.10], dtype=np.float32)
                    wrist_forward = _quat_rotate_vec_wxyz(ee_ori, np.asarray([0.30, 0.0, -0.05], dtype=np.float32))
                    wrist_target = 0.35 * (ee_pos + wrist_forward) + 0.65 * cloth_right_focus
                    cameras["right_wrist"].set_world_pose(
                        position=cam_pos,
                        orientation=_look_at_quat_wxyz(camera_pos=cam_pos, target_pos=wrist_target),
                    )
                    if int(getattr(args, "print_cam_every", 0)) > 0 and (step % int(args.print_cam_every) == 0):
                        print(
                            f"[cam] right_wrist_pos={cam_pos.tolist()} ee_pos={ee_pos.tolist()} "
                            f"look_at={wrist_target.tolist()}"
                        )

                # Step a few frames to ensure sensors update
                for _ in range(max(1, int(getattr(args, "camera_warmup_steps", 5)))):
                    world.step(render=True)

                read_attempts = max(1, int(getattr(args, "camera_read_attempts", 8)))
                overview_rgb = _read_camera_rgb(cameras["overview"], world=world, max_attempts=read_attempts)
                left_ext_rgb = _read_camera_rgb(cameras["left_exterior"], world=world, max_attempts=read_attempts)
                right_ext_rgb = _read_camera_rgb(cameras["right_exterior"], world=world, max_attempts=read_attempts)
                left_rgb = _read_camera_rgb(cameras["left_wrist"], world=world, max_attempts=read_attempts)
                right_rgb = _read_camera_rgb(cameras["right_wrist"], world=world, max_attempts=read_attempts)

                # Build inputs for each arm (DROID-style)
                jL, gL = _arm_state_from_articulation(arm_left)
                jR, gR = _arm_state_from_articulation(arm_right)
                target_left_pos, target_right_pos, target_gripper_open, phase_name = _fold_targets_for_step(
                    step=int(step),
                    cloth_center=cloth_center,
                )

                prev_memory = language_memory
                hl_raw = None
                raw_actions_left = None
                raw_actions_right = None
                structured_state: dict[str, Any] = {}
                retrieved_semantic_summary = ""
                retrieved_visual_summary = ""
                retrieved_semantic_hint = ""
                retrieved_visual_hint = ""
                pcmb_debug: dict[str, Any] = {}

                if rpc is not None:
                    resp = rpc.call(
                        {
                            "cmd": "step",
                            "step": int(step),
                            "base_rgb": overview_rgb,
                            "left_exterior_rgb": left_ext_rgb,
                            "right_exterior_rgb": right_ext_rgb,
                            "left_wrist_rgb": left_rgb,
                            "right_wrist_rgb": right_rgb,
                            "jL": jL,
                            "gL": gL,
                            "jR": jR,
                            "gR": gR,
                            "num_steps": int(args.num_steps),
                        }
                    )

                    raw_actions_left = np.asarray(resp["actions_left"], dtype=np.float32)
                    raw_actions_right = np.asarray(resp["actions_right"], dtype=np.float32)
                    actions_both = np.asarray(
                        resp.get("actions_both", np.concatenate([raw_actions_left, raw_actions_right], axis=1)),
                        dtype=np.float32,
                    )

                    language_memory = str(resp.get("language_memory", language_memory))
                    subtask = str(resp.get("subtask", phase_name)).strip() or phase_name
                    subtask = merge_subtask_with_phase(goal=goal, subtask=subtask, phase_name=phase_name)
                    low_level_prompt = str(resp.get("low_level_prompt", "")).strip()
                    hl_raw = resp.get("hl_raw_text")
                    structured_state = dict(resp.get("structured_state") or {})
                    retrieved_semantic_summary = str(resp.get("retrieved_semantic_summary", ""))
                    retrieved_visual_summary = str(resp.get("retrieved_visual_summary", ""))
                    retrieved_semantic_hint = str(resp.get("retrieved_semantic_hint", ""))
                    retrieved_visual_hint = str(resp.get("retrieved_visual_hint", ""))
                    pcmb_debug = dict(resp.get("pcmb_debug") or {})
                else:
                    # Long-term memory update (single, shared for both arms)
                    subtask = phase_name
                    if hl is not None:
                        hl_result = hl.update(
                            goal=goal,
                            prev_memory=language_memory,
                            image=overview_rgb,
                            phase_name=phase_name,
                            step=int(step),
                        )
                        language_memory = hl_result.memory
                        subtask = str(hl_result.subtask).strip() or phase_name
                        subtask = merge_subtask_with_phase(goal=goal, subtask=subtask, phase_name=phase_name)
                        hl_raw = hl_result.raw_text
                        structured_state = dict(hl_result.structured_state)
                        retrieved_semantic_summary = str(hl_result.retrieved_semantic_summary)
                        retrieved_visual_summary = str(hl_result.retrieved_visual_summary)
                        retrieved_semantic_hint = str(hl_result.retrieved_semantic_hint)
                        retrieved_visual_hint = str(hl_result.retrieved_visual_hint)
                        pcmb_debug = dict(hl_result.pcmb_debug)
                    elif "fold" in goal.lower() and "shirt" in goal.lower():
                        language_memory = (
                            f"Current phase: {phase_name}. Two arms should stay low over the shirt and move toward the center fold."
                        )

                    low_level_prompt = "\n".join(
                        line
                        for line in [
                            f"Task: {goal}",
                            f"Phase: {phase_name}",
                            f"Subtask: {subtask}",
                            (
                                "State: " + "; ".join(f"{k}={v}" for k, v in structured_state.items() if v)
                                if structured_state
                                else ""
                            ),
                            f"Retrieved semantic cue: {retrieved_semantic_hint}" if retrieved_semantic_hint else "",
                            f"Retrieved visual cue: {retrieved_visual_hint}" if retrieved_visual_hint else "",
                            f"Target left TCP: {np.round(target_left_pos, 3).tolist()}",
                            f"Target right TCP: {np.round(target_right_pos, 3).tolist()}",
                            "Coordinate both arms to create a clean shirt fold.",
                        ]
                        if line
                    ).strip()

                    stack_left.push("base", left_ext_rgb)
                    stack_left.push("wrist", left_rgb)
                    stack_right.push("base", right_ext_rgb)
                    stack_right.push("wrist", right_rgb)

                    low_level_prompt_left = compose_low_level_prompt(
                        goal=goal,
                        subtask=subtask,
                        language_memory=language_memory,
                        arm_side="left",
                        phase_name=phase_name,
                        structured_state_summary="; ".join(f"{k}={v}" for k, v in structured_state.items() if v),
                        retrieved_semantic_hint=retrieved_semantic_hint,
                        retrieved_visual_hint=retrieved_visual_hint,
                    )
                    low_level_prompt_right = compose_low_level_prompt(
                        goal=goal,
                        subtask=subtask,
                        language_memory=language_memory,
                        arm_side="right",
                        phase_name=phase_name,
                        structured_state_summary="; ".join(f"{k}={v}" for k, v in structured_state.items() if v),
                        retrieved_semantic_hint=retrieved_semantic_hint,
                        retrieved_visual_hint=retrieved_visual_hint,
                    )
                    ex_left = {
                        "observation/exterior_image_1_left": stack_left.get("base"),
                        "observation/wrist_image_left": stack_left.get("wrist"),
                        "observation/joint_position": jL,
                        "observation/gripper_position": gL,
                        "prompt": low_level_prompt_left,
                    }
                    ex_right = {
                        "observation/exterior_image_1_left": stack_right.get("base"),
                        "observation/wrist_image_left": stack_right.get("wrist"),
                        "observation/joint_position": jR,
                        "observation/gripper_position": gR,
                        "prompt": low_level_prompt_right,
                    }

                    if policy is None:
                        raw_actions_left, raw_actions_right, _ = _scripted_fold_actions(
                            step=int(step),
                            horizon=max(1, int(getattr(args, "execute_horizon", 0)) or 8),
                        )
                    else:
                        out_left = policy.infer(ex_left, num_steps=int(args.num_steps))
                        out_right = policy.infer(ex_right, num_steps=int(args.num_steps))
                        raw_actions_left = np.asarray(out_left["actions"], dtype=np.float32)  # [H,8]
                        raw_actions_right = np.asarray(out_right["actions"], dtype=np.float32)  # [H,8]

                scripted_left, scripted_right, _phase_name = _scripted_fold_actions(
                    step=int(step),
                    horizon=max(
                        1,
                        int(
                            max(
                                raw_actions_left.shape[0] if raw_actions_left is not None else 0,
                                raw_actions_right.shape[0] if raw_actions_right is not None else 0,
                                int(getattr(args, "execute_horizon", 0)) or 8,
                            )
                        ),
                    ),
                )
                residual_scale = float(getattr(args, "task_adapter_residual_scale", 0.20))
                if raw_actions_left is not None:
                    h_left = min(scripted_left.shape[0], raw_actions_left.shape[0])
                    actions_left = scripted_left[:h_left] + residual_scale * np.clip(raw_actions_left[:h_left], -1.0, 1.0)
                else:
                    actions_left = scripted_left
                if raw_actions_right is not None:
                    h_right = min(scripted_right.shape[0], raw_actions_right.shape[0])
                    actions_right = scripted_right[:h_right] + residual_scale * np.clip(raw_actions_right[:h_right], -1.0, 1.0)
                else:
                    actions_right = scripted_right
                min_h = min(actions_left.shape[0], actions_right.shape[0])
                actions_left = np.clip(actions_left[:min_h], -1.0, 1.0)
                actions_right = np.clip(actions_right[:min_h], -1.0, 1.0)
                actions_both = np.concatenate([actions_left, actions_right], axis=1)  # [H,16]
                last_actions_both = actions_both

                if writer is not None and rpc is None:
                    writer.add_step(
                        step,
                        goal=goal,
                        low_level_prompt=low_level_prompt,
                        prev_memory=prev_memory,
                        language_memory=language_memory,
                        subtask=subtask,
                        hl_raw_text=hl_raw,
                        structured_state=structured_state,
                        retrieved_semantic_summary=retrieved_semantic_summary,
                        retrieved_visual_summary=retrieved_visual_summary,
                        pcmb_debug=pcmb_debug,
                        images={
                            "overview": overview_rgb,
                            "left_exterior": stack_left.get("base"),
                            "right_exterior": stack_right.get("base"),
                            "left_wrist": stack_left.get("wrist"),
                            "right_wrist": stack_right.get("wrist"),
                        },
                        actions=actions_both,
                    )

                print(f"[step {step}] subtask: {subtask}")
                print(f"[step {step}] language_memory_len: {len(language_memory)}")
                print(f"[step {step}] actions_left: {actions_left.shape} actions_right: {actions_right.shape}")

                # Apply the action chunk to both arms using cloth-centric IK targets with pi0.5 residuals.
                horizon = min(actions_left.shape[0], actions_right.shape[0])
                exec_h = int(getattr(args, "execute_horizon", 0))
                if exec_h > 0:
                    horizon = min(horizon, exec_h)
                for t in range(horizon):
                    interp = float(t + 1) / float(max(1, horizon))
                    left_goal = cloth_center + interp * (target_left_pos - cloth_center)
                    right_goal = cloth_center + interp * (target_right_pos - cloth_center)
                    left_joint_targets = _solve_franka_ik_targets(
                        arm_left,
                        ee_index=ee_left,
                        goal_position=left_goal,
                        residual_action=actions_left[t],
                        residual_scale=float(getattr(args, "task_adapter_residual_scale", 0.20)),
                    )
                    right_joint_targets = _solve_franka_ik_targets(
                        arm_right,
                        ee_index=ee_right,
                        goal_position=right_goal,
                        residual_action=actions_right[t],
                        residual_scale=float(getattr(args, "task_adapter_residual_scale", 0.20)),
                    )
                    if left_joint_targets is not None:
                        _set_arm_joint_targets(
                            arm_left,
                            joint_targets_7=left_joint_targets,
                            gripper_open=target_gripper_open,
                        )
                    else:
                        _apply_action_to_franka(
                            arm_left,
                            actions_left[t],
                            joint_delta_scale=float(args.joint_delta_scale),
                            gripper_delta_scale=float(args.gripper_delta_scale),
                            control_dt=control_dt,
                        )
                    if right_joint_targets is not None:
                        _set_arm_joint_targets(
                            arm_right,
                            joint_targets_7=right_joint_targets,
                            gripper_open=target_gripper_open,
                        )
                    else:
                        _apply_action_to_franka(
                            arm_right,
                            actions_right[t],
                            joint_delta_scale=float(args.joint_delta_scale),
                            gripper_delta_scale=float(args.gripper_delta_scale),
                            control_dt=control_dt,
                        )
                    for _ in range(int(args.sim_steps_per_action)):
                        world.step(render=True)

                if (not bool(args.no_cloth)) and (int(step) % 4 == 0):
                    _update_shirt_fold_visual(
                        stage=stage,
                        cloth_root_path=cloth_root_path,
                        step=int(step),
                        cloth_center=np.asarray([0.6, 0.0, float(args.cloth_z)], dtype=np.float32),
                    )

                step += 1
        except KeyboardInterrupt:
            print("[main] interrupted; finalizing...")

        if writer is not None:
            writer.finalize(final_actions=last_actions_both if last_actions_both is not None else None)
            (writer.run_dir / "final_language_memory.txt").write_text(language_memory, encoding="utf-8")
            (writer.run_dir / "summary.json").write_text(
                json.dumps({"goal": goal, "final_subtask": subtask, "final_memory_len": len(language_memory)}, indent=2)
                + "\n",
                encoding="utf-8",
            )

        if rpc is not None:
            try:
                try:
                    rpc.call({"cmd": "finalize"})
                except Exception:
                    pass
                rpc.close()
                if bool(args.rpc_close_on_exit):
                    rpc = RpcClient(
                        host=str(args.rpc_host),
                        port=int(args.rpc_port),
                        timeout_s=float(args.rpc_timeout_s),
                    )
                    try:
                        rpc.call({"cmd": "close"})
                    finally:
                        rpc.close()
            finally:
                pass
        if rpc_server_proc is not None:
            if bool(args.rpc_close_on_exit):
                try:
                    rpc_server_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    rpc_server_proc.terminate()
                    try:
                        rpc_server_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        rpc_server_proc.kill()
            else:
                print(f"[rpc] leaving warm server running (pid={rpc_server_proc.pid})")

        if not bool(args.headless) and bool(args.stay_open):
            print("[gui] stay_open enabled; close the Isaac Sim window to exit.")
            try:
                while simulation_app.is_running():
                    world.step(render=True)
            except KeyboardInterrupt:
                pass
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
