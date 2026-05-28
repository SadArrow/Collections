#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE", "/root/workspace/qianyupeng"))
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_ISAAC_PY = DEFAULT_BASE / "isaac-sim-standalone@4.5.0" / "python.sh"
DEFAULT_RUNTIME_DIR = DEFAULT_BASE / "myVLA" / "WorldModelDiffusionVlaRuntime"
DEFAULT_WRAPPER = DEFAULT_BASE / "myVLA" / "scripts" / "server" / "env_validation_entry_wrapper.py"
DEFAULT_CARBONITE_SEMAPHORE = Path("/dev/shm/sem.carbonite-sharedmemory")
DEFAULT_EXTRA_PYDEPS = DEFAULT_RUNTIME_DIR / "pydeps"
DEFAULT_VK_ICD_JSON = DEFAULT_BASE / "vulkan_test" / "nvidia_abs_egl.json"
DEFAULT_DOWNLOADS_VK_ICD_JSON = DEFAULT_BASE / "downloads" / "nvidia570" / "nvidia_egl_icd_570.86.10.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DexGarmentLab Env_StandAlone-style entry with the stable Isaac server wrapper."
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id isolated via CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--gpu_binding_mode",
        choices=("cuda_visible_devices", "omniverse"),
        default="cuda_visible_devices",
        help="Use CUDA_VISIBLE_DEVICES isolation or direct Omniverse physical GPU selection.",
    )
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--runtime_dir", default=os.fspath(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--extra_pydeps", default=os.fspath(DEFAULT_EXTRA_PYDEPS))
    parser.add_argument("--wrapper_script", default=os.fspath(DEFAULT_WRAPPER))
    parser.add_argument("--carbonite_semaphore", default=os.fspath(DEFAULT_CARBONITE_SEMAPHORE))
    parser.add_argument("--vk_icd_json", default=os.fspath(DEFAULT_VK_ICD_JSON))
    parser.add_argument("--script_rel", default="tools/myvla_fold_tops_envstandalone_entry.py")
    parser.add_argument("--portable_root", default="")
    parser.add_argument(
        "--headless_excluded_extensions",
        default="isaacsim.asset.importer.urdf,isaacsim.asset.importer.mjcf",
    )
    parser.add_argument("--disable_fabric_delegate", action="store_true")
    parser.add_argument(
        "--run_as_main",
        action="store_true",
        help="Forward --run_as_main to env_validation_entry_wrapper so __main__ blocks execute.",
    )
    parser.add_argument("--verbose_wrapper", action="store_true")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the wrapped target.")
    return parser.parse_args()


def _safe_symlink(*, link_path: Path, candidates: list[Path]) -> None:
    if link_path.exists():
        return
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(resolved)
            return


def _build_extcache_library_path(isaac_root: Path) -> str:
    extscache_root = isaac_root / "extscache"
    if not extscache_root.is_dir():
        return ""
    items: list[str] = []
    seen: set[str] = set()
    for pattern in ("*/bin", "*/lib", "*/bin/deps", "*/lib/deps"):
        for path in sorted(extscache_root.glob(pattern)):
            if not path.is_dir():
                continue
            text = os.fspath(path)
            if text in seen:
                continue
            seen.add(text)
            items.append(text)
    return ":".join(items)


def _first_existing_dir(paths: list[Path]) -> Path | None:
    for candidate in paths:
        if candidate.is_dir():
            return candidate
    return None


def _first_existing_file(paths: list[Path]) -> Path | None:
    for candidate in paths:
        if candidate.is_file():
            return candidate
    return None


def _first_dir_with_marker(paths: list[Path], marker_name: str) -> Path | None:
    for candidate in paths:
        if candidate.is_dir() and (candidate / marker_name).exists():
            return candidate
    return None


def _version_key_from_suffix(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(text).split("."):
        if not token.isdigit():
            return tuple()
        parts.append(int(token))
    return tuple(parts)


def _latest_versioned_file(search_roots: list[Path], prefix: str) -> Path | None:
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for candidate in root.glob(f"{prefix}*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            suffix = candidate.name[len(prefix) :]
            version_key = _version_key_from_suffix(suffix)
            if not version_key:
                continue
            candidates.append((version_key, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    extra_pydeps = Path(args.extra_pydeps).expanduser().resolve()
    local_syslib_dir = runtime_dir / "syslibs" / "usr" / "lib" / "x86_64-linux-gnu"
    local_syslib_dir.mkdir(parents=True, exist_ok=True)
    isaac_root = Path(args.isaac_python).expanduser().resolve().parents[0]
    base = Path(args.dex_root).expanduser().resolve().parents[0]

    clean_nvidia_fix_root = base / "downloads" / "nvidia570_fix" / "extract"
    clean_nvidia_gl_root = clean_nvidia_fix_root / "libnvidia-gl-570"
    clean_nvidia_compute_root = clean_nvidia_fix_root / "libnvidia-compute-570"
    clean_nvidia_cfg_root = clean_nvidia_fix_root / "libnvidia-cfg1-570"
    downloaded_nvidia_gl_root = base / "downloads" / "nvidia570" / "libnvidia-gl-570_570.86.10-0ubuntu1_amd64"
    clean_nvidia_gl_lib_dir = clean_nvidia_gl_root / "usr" / "lib" / "x86_64-linux-gnu"
    clean_nvidia_compute_lib_dir = clean_nvidia_compute_root / "usr" / "lib" / "x86_64-linux-gnu"
    clean_nvidia_cfg_lib_dir = clean_nvidia_cfg_root / "usr" / "lib" / "x86_64-linux-gnu"
    downloaded_nvidia_gl_lib_dir = downloaded_nvidia_gl_root / "usr" / "lib" / "x86_64-linux-gnu"
    downloaded_nvidia_egl_vendor_dir = downloaded_nvidia_gl_root / "usr" / "share" / "glvnd" / "egl_vendor.d"
    downloaded_nvidia_egl_platform_dir = downloaded_nvidia_gl_root / "usr" / "share" / "egl" / "egl_external_platform.d"
    downloaded_nvidia_vk_layer_dir = downloaded_nvidia_gl_root / "usr" / "share" / "vulkan" / "implicit_layer.d"
    clean_nvidia_egl_vendor_dir = clean_nvidia_gl_root / "usr" / "share" / "glvnd" / "egl_vendor.d"
    clean_nvidia_egl_platform_dir = clean_nvidia_gl_root / "usr" / "share" / "egl" / "egl_external_platform.d"
    clean_nvidia_vk_layer_dir = clean_nvidia_gl_root / "usr" / "share" / "vulkan" / "implicit_layer.d"
    clean_nvidia_vk_icd_json = base / "downloads" / "nvidia570_fix" / "nvidia_egl_icd_570.133.07.json"
    if clean_nvidia_gl_lib_dir.is_dir():
        clean_nvidia_vk_icd_json.parent.mkdir(parents=True, exist_ok=True)
        clean_nvidia_vk_icd_json.write_text(
            "{\n"
            '  "file_format_version": "1.0.1",\n'
            '  "ICD": {\n'
            f'    "library_path": "{clean_nvidia_gl_lib_dir / "libEGL_nvidia.so.0"}",\n'
            '    "api_version": "1.4.303"\n'
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

    system_nvidia_gl_lib_dir = _first_dir_with_marker(
        [
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/lib/x86_64-linux-gnu"),
        ],
        "libEGL_nvidia.so.0",
    )
    # Prefer the historically working 570.86.10 workspace bundle over the
    # later 570.133.07 "fix" bundle. The A6000 host currently reports driver
    # 570.86.10, and mixing a newer Vulkan/EGL user-space stack with that
    # kernel driver caused vkCreateInstance(ERROR_INCOMPATIBLE_DRIVER).
    nvidia_gl_lib_dir = (
        downloaded_nvidia_gl_lib_dir
        if downloaded_nvidia_gl_lib_dir.is_dir()
        else clean_nvidia_gl_lib_dir
        if clean_nvidia_gl_lib_dir.is_dir()
        else system_nvidia_gl_lib_dir
    )
    # Keep compute/cfg resolution aligned with the system driver first; only
    # fall back to extracted bundles if the matching system libraries are absent.
    nvidia_compute_lib_dir = None
    nvidia_cfg_lib_dir = None
    egl_vendor_dir = _first_dir_with_marker(
        [
            downloaded_nvidia_egl_vendor_dir,
            clean_nvidia_egl_vendor_dir,
            Path("/usr/share/glvnd/egl_vendor.d"),
            Path("/lib/glvnd/egl_vendor.d"),
        ],
        "10_nvidia.json",
    ) or downloaded_nvidia_egl_vendor_dir
    egl_platform_dir = _first_existing_dir(
        [
            downloaded_nvidia_egl_platform_dir,
            clean_nvidia_egl_platform_dir,
            Path("/usr/share/egl/egl_external_platform.d"),
            Path("/lib/egl/egl_external_platform.d"),
            downloaded_nvidia_egl_platform_dir,
        ]
    )
    vk_layer_dir = _first_dir_with_marker(
        [
            downloaded_nvidia_vk_layer_dir,
            clean_nvidia_vk_layer_dir,
            Path("/usr/share/vulkan/implicit_layer.d"),
            Path("/etc/vulkan/implicit_layer.d"),
        ],
        "nvidia_layers.json",
    ) or downloaded_nvidia_vk_layer_dir
    nvidia_decode_lib_dir = _first_existing_dir(
        sorted(
            (base / "downloads" / "nvidia570").glob("libnvidia-decode-*"),
            key=lambda item: item.name,
            reverse=True,
        )
    )
    if nvidia_decode_lib_dir is not None:
        nvidia_decode_lib_dir = nvidia_decode_lib_dir / "usr" / "lib" / "x86_64-linux-gnu"
    x11_runtime_lib_dir = base / "downloads" / "x11_runtime_libs" / "usr" / "lib" / "x86_64-linux-gnu"
    libxt_lib_dir = base / "downloads" / "libxt6" / "usr" / "lib" / "x86_64-linux-gnu"

    _safe_symlink(
        link_path=local_syslib_dir / "libcuda.so",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so"),
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
            *( [clean_nvidia_compute_lib_dir / "libcuda.so"] if clean_nvidia_compute_lib_dir.is_dir() else [] ),
        ],
    )
    _safe_symlink(
        link_path=local_syslib_dir / "libcuda.so.1",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
            *( [clean_nvidia_compute_lib_dir / "libcuda.so.1"] if clean_nvidia_compute_lib_dir.is_dir() else [] ),
            local_syslib_dir / "libcuda.so",
        ],
    )
    _safe_symlink(
        link_path=local_syslib_dir / "libGLU.so.1",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libGLU.so.1"),
            Path("/lib/x86_64-linux-gnu/libGLU.so.1"),
        ],
    )
    latest_nvidia_allocator = _latest_versioned_file(
        [downloaded_nvidia_gl_lib_dir, Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"), clean_nvidia_gl_lib_dir],
        "libnvidia-allocator.so.",
    )
    if latest_nvidia_allocator is not None:
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-allocator.so.1",
            candidates=[latest_nvidia_allocator],
        )
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-allocator.so",
            candidates=[latest_nvidia_allocator],
        )
    latest_nvidia_ml = _latest_versioned_file(
        [Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"), clean_nvidia_compute_lib_dir],
        "libnvidia-ml.so.",
    )
    if latest_nvidia_ml is not None:
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-ml.so.1",
            candidates=[latest_nvidia_ml],
        )
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-ml.so",
            candidates=[latest_nvidia_ml],
        )
    latest_nvidia_cfg = _latest_versioned_file(
        [Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"), clean_nvidia_cfg_lib_dir],
        "libnvidia-cfg.so.",
    )
    if latest_nvidia_cfg is not None:
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-cfg.so.1",
            candidates=[latest_nvidia_cfg],
        )
        _safe_symlink(
            link_path=local_syslib_dir / "libnvidia-cfg.so",
            candidates=[latest_nvidia_cfg],
        )
    nvcuvid_so1 = _first_existing_file(
        [
            Path("/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1"),
            Path("/lib/x86_64-linux-gnu/libnvcuvid.so.1"),
            *(
                [nvidia_decode_lib_dir / "libnvcuvid.so.1"]
                if nvidia_decode_lib_dir is not None
                else []
            ),
        ]
    )
    if nvcuvid_so1 is not None:
        _safe_symlink(link_path=local_syslib_dir / "libnvcuvid.so.1", candidates=[nvcuvid_so1])
        _safe_symlink(link_path=local_syslib_dir / "libnvcuvid.so", candidates=[nvcuvid_so1])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    use_cvd = str(args.gpu_binding_mode) == "cuda_visible_devices"
    if use_cvd:
        env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    active_gpu = 0 if use_cvd else int(args.gpu)
    env["DEXGARMENTLAB_ACTIVE_GPU"] = str(int(active_gpu))
    env["DEXGARMENTLAB_PHYSICS_GPU"] = str(int(active_gpu))

    ld_parts = [os.fspath(local_syslib_dir)]
    for candidate in (
        x11_runtime_lib_dir,
        libxt_lib_dir,
        nvidia_gl_lib_dir,
        nvidia_compute_lib_dir,
        nvidia_cfg_lib_dir,
        nvidia_decode_lib_dir,
    ):
        if candidate is None:
            continue
        if candidate.is_dir():
            ld_parts.append(os.fspath(candidate))
    ld_parts.extend(["/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"])
    extscache_paths = _build_extcache_library_path(isaac_root)
    if extscache_paths:
        ld_parts.extend([item for item in extscache_paths.split(":") if item])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts + [str(env.get("LD_LIBRARY_PATH", ""))]).rstrip(":")

    requested_vk_icd = Path(args.vk_icd_json).expanduser().resolve() if str(args.vk_icd_json).strip() else Path()
    system_vk_icd_json = _first_existing_file(
        [
            Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
            Path("/etc/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"),
        ]
    )
    use_default_vk_icd = requested_vk_icd == DEFAULT_VK_ICD_JSON.resolve()
    if use_default_vk_icd and DEFAULT_DOWNLOADS_VK_ICD_JSON.is_file():
        vk_icd_json = DEFAULT_DOWNLOADS_VK_ICD_JSON.resolve()
    elif use_default_vk_icd and clean_nvidia_vk_icd_json.is_file():
        vk_icd_json = clean_nvidia_vk_icd_json.resolve()
    elif use_default_vk_icd and DEFAULT_VK_ICD_JSON.is_file():
        # Keep the previously working Isaac path: a minimal EGL-backed ICD JSON.
        vk_icd_json = DEFAULT_VK_ICD_JSON.resolve()
    elif use_default_vk_icd and system_vk_icd_json is not None:
        vk_icd_json = system_vk_icd_json.resolve()
    else:
        vk_icd_json = requested_vk_icd
    if (not str(args.vk_icd_json).strip() or not vk_icd_json.is_file()) and DEFAULT_DOWNLOADS_VK_ICD_JSON.is_file():
        vk_icd_json = DEFAULT_DOWNLOADS_VK_ICD_JSON.resolve()
    if vk_icd_json and vk_icd_json.is_file():
        env["VK_ICD_FILENAMES"] = os.fspath(vk_icd_json)
        env["VK_DRIVER_FILES"] = os.fspath(vk_icd_json)
    if egl_vendor_dir.is_dir():
        env["__EGL_VENDOR_LIBRARY_DIRS"] = os.fspath(egl_vendor_dir)
    if egl_platform_dir.is_dir():
        env["__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS"] = os.fspath(egl_platform_dir)
    if vk_layer_dir.is_dir():
        env["VK_LAYER_PATH"] = os.fspath(vk_layer_dir)
    if extra_pydeps.is_dir():
        env["ENV_VALIDATION_EXTRA_PYTHONPATH"] = os.fspath(extra_pydeps)
    return env


def _clear_carbonite_semaphore(path_text: str) -> str:
    path = Path(str(path_text).strip())
    if not str(path):
        return ""
    try:
        if path.exists():
            path.unlink()
            return f"cleared:{os.fspath(path)}"
        return f"missing:{os.fspath(path)}"
    except Exception as exc:
        return f"clear_failed:{os.fspath(path)}:{type(exc).__name__}:{exc}"


def _clean_target_args(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return list(raw)


def _default_portable_root(args: argparse.Namespace) -> Path:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    return runtime_dir / "wrapped_envstandalone" / f"gpu{int(args.gpu)}"


def main() -> int:
    args = _parse_args()
    dex_root = Path(args.dex_root).expanduser().resolve()
    wrapper_script = Path(args.wrapper_script).expanduser().resolve()
    isaac_python = Path(args.isaac_python).expanduser().resolve()
    portable_root = (
        Path(args.portable_root).expanduser().resolve()
        if str(args.portable_root).strip()
        else _default_portable_root(args)
    )
    portable_root.mkdir(parents=True, exist_ok=True)

    env = _build_env(args)
    semaphore_status = _clear_carbonite_semaphore(str(args.carbonite_semaphore))
    forwarded_args = _clean_target_args(args.target_args)

    use_cvd = str(args.gpu_binding_mode) == "cuda_visible_devices"
    active_gpu = 0 if use_cvd else int(args.gpu)

    command = [
        os.fspath(isaac_python),
        os.fspath(wrapper_script),
        "--dex_root",
        os.fspath(dex_root),
        "--script_rel",
        str(args.script_rel).replace("\\", "/"),
        "--portable_root",
        os.fspath(portable_root),
        "--active_gpu",
        str(int(active_gpu)),
        "--physics_gpu",
        str(int(active_gpu)),
        "--headless_excluded_extensions",
        str(args.headless_excluded_extensions),
        "--call_main_if_present",
    ]
    if bool(args.disable_fabric_delegate):
        command.append("--disable_fabric_delegate")
    if bool(args.run_as_main):
        command.append("--run_as_main")
    if bool(args.verbose_wrapper):
        command.append("--verbose")
    command.extend(["--"] + forwarded_args)

    print(f"[launcher] dex_root={dex_root}", flush=True)
    print(f"[launcher] isaac_python={isaac_python}", flush=True)
    print(f"[launcher] wrapper_script={wrapper_script}", flush=True)
    print(f"[launcher] portable_root={portable_root}", flush=True)
    print(f"[launcher] gpu={int(args.gpu)}", flush=True)
    print(f"[launcher] gpu_binding_mode={args.gpu_binding_mode}", flush=True)
    print(f"[launcher] env_cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"[launcher] env_dex_active_gpu={env.get('DEXGARMENTLAB_ACTIVE_GPU', '')}", flush=True)
    print(f"[launcher] env_dex_physics_gpu={env.get('DEXGARMENTLAB_PHYSICS_GPU', '')}", flush=True)
    print(f"[launcher] env_vk_icd={env.get('VK_ICD_FILENAMES', '')}", flush=True)
    print(f"[launcher] carbonite_semaphore={semaphore_status}", flush=True)
    print(f"[launcher] command={' '.join(command)}", flush=True)

    proc = subprocess.run(
        command,
        cwd=os.fspath(dex_root),
        env=env,
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
