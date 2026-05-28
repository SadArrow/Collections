#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a DexGarmentLab Env_Validation entrypoint without modifying the source file."
    )
    parser.add_argument("--dex_root", required=True, help="DexGarmentLab repository root on the server.")
    parser.add_argument("--script_rel", required=True, help="Relative path under dex_root, e.g. Env_Validation/Fold_Tops_HALO.py")
    parser.add_argument("--portable_root", default="", help="Portable root forwarded to SimulationApp extra_args.")
    parser.add_argument("--active_gpu", type=int, default=-1, help="Renderer GPU id inside the visible device set.")
    parser.add_argument("--physics_gpu", type=int, default=-1, help="Physics GPU id inside the visible device set.")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable SimulationApp multi_gpu.")
    parser.add_argument("--disable_fabric_delegate", action="store_true")
    parser.add_argument(
        "--headless_excluded_extensions",
        default="isaacsim.asset.importer.urdf,isaacsim.asset.importer.mjcf",
        help="Comma-separated extension ids excluded from headless startup.",
    )
    parser.add_argument(
        "--disable_open3d_ml_stub",
        action="store_true",
        help="Do not install the lightweight open3d.ml stub before importing the target script.",
    )
    parser.add_argument(
        "--call_main_if_present",
        action="store_true",
        help="After importing the target module, call module.main() when it exists.",
    )
    parser.add_argument(
        "--run_as_main",
        action="store_true",
        help="Execute the target script with runpy.run_path(..., run_name='__main__') so __main__ blocks run.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target Env_Validation script.")
    return parser.parse_args()


def _clean_target_args(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return list(raw)


def _merge_extra_arg(extra_args: list[str], item: str) -> None:
    if item not in extra_args:
        extra_args.append(item)


def _patch_simulation_app(args: argparse.Namespace) -> None:
    import isaacsim

    original_simulation_app = isaacsim.SimulationApp
    excluded_extensions = [item.strip() for item in str(args.headless_excluded_extensions).split(",") if item.strip()]
    portable_root = str(args.portable_root).strip()
    disable_fabric_delegate = bool(args.disable_fabric_delegate)
    active_gpu = int(args.active_gpu)
    physics_gpu = int(args.physics_gpu)
    multi_gpu = bool(args.multi_gpu)
    verbose = bool(args.verbose)

    class WrappedSimulationApp:
        def __init__(self, launch_config: dict[str, Any] | None = None) -> None:
            config = dict(launch_config or {})
            if multi_gpu:
                config["multi_gpu"] = True
            if bool(config.get("headless", False)):
                config.setdefault("hide_ui", True)
            extra_args = list(config.get("extra_args", []))
            if disable_fabric_delegate:
                _merge_extra_arg(extra_args, "--/app/useFabricSceneDelegate=0")
            # Omniverse/Isaac Sim official workaround for newer Vulkan driver versions
            # being reported as incompatible on Linux.
            _merge_extra_arg(extra_args, "--/rtx/verifyDriverVersion/enabled=false")
            if active_gpu >= 0:
                _merge_extra_arg(extra_args, f"--/renderer/activeGpu={active_gpu}")
            if physics_gpu >= 0:
                _merge_extra_arg(extra_args, f"--/physics/cudaDevice={physics_gpu}")
            if portable_root:
                _merge_extra_arg(extra_args, "--portable-root")
                _merge_extra_arg(extra_args, portable_root)
            if bool(config.get("headless", False)) and excluded_extensions:
                _merge_extra_arg(extra_args, f"--/app/extensions/excluded={json.dumps(excluded_extensions)}")
            if extra_args:
                config["extra_args"] = extra_args
            if verbose:
                print(
                    "[env-wrapper] patched SimulationApp config:",
                    json.dumps(
                        {
                            "headless": bool(config.get("headless", False)),
                            "hide_ui": bool(config.get("hide_ui", False)),
                            "multi_gpu": bool(config.get("multi_gpu", False)),
                            "extra_args": list(config.get("extra_args", [])),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            self._app = original_simulation_app(config)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._app, name)

    isaacsim.SimulationApp = WrappedSimulationApp


def _install_open3d_ml_stub(*, enabled: bool, verbose: bool) -> None:
    if not enabled:
        return
    if "open3d.ml" in sys.modules:
        return
    stub = types.ModuleType("open3d.ml")
    stub.__dict__.update(
        {
            "__file__": "<env_validation_wrapper_stub>",
            "__package__": "open3d",
            "__path__": [],
            "__all__": [],
        }
    )
    sys.modules["open3d.ml"] = stub
    if verbose:
        print("[env-wrapper] installed open3d.ml stub", flush=True)


def _extend_sys_path_from_env(*, verbose: bool) -> None:
    raw = os.environ.get("ENV_VALIDATION_EXTRA_PYTHONPATH", "").strip()
    if not raw:
        return
    added: list[str] = []
    for item in raw.split(os.pathsep):
        text = item.strip()
        if not text:
            continue
        path = os.fspath(Path(text).expanduser().resolve())
        if path in sys.path or not Path(path).exists():
            continue
        sys.path.append(path)
        added.append(path)
    if verbose and added:
        print(
            "[env-wrapper] appended extra python paths:",
            json.dumps(added, ensure_ascii=False),
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    dex_root = Path(args.dex_root).expanduser().resolve()
    target_path = dex_root / str(args.script_rel)
    if not target_path.is_file():
        raise FileNotFoundError(f"Target Env_Validation script not found: {target_path}")

    portable_root = Path(args.portable_root).expanduser().resolve() if str(args.portable_root).strip() else None
    if portable_root is not None:
        (portable_root / "cache").mkdir(parents=True, exist_ok=True)
        (portable_root / "data").mkdir(parents=True, exist_ok=True)
        (portable_root / "logs").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("OMNI_CACHE_DIR", os.fspath(portable_root / "cache"))
        os.environ.setdefault("OMNI_DATA_DIR", os.fspath(portable_root / "data"))
        os.environ.setdefault("OMNI_LOGS_DIR", os.fspath(portable_root / "logs"))
        os.environ.setdefault("OMNI_USER_DIR", os.fspath(portable_root))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("DEXGARMENTLAB_HEADLESS", "1")
    os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")
    if int(args.active_gpu) >= 0:
        os.environ["DEXGARMENTLAB_ACTIVE_GPU"] = str(int(args.active_gpu))
    if int(args.physics_gpu) >= 0:
        os.environ["DEXGARMENTLAB_PHYSICS_GPU"] = str(int(args.physics_gpu))
    if bool(args.multi_gpu):
        os.environ["DEXGARMENTLAB_MULTI_GPU"] = "1"
    os.environ["MYVLA_ENV_VALIDATION_WRAPPER_ACTIVE"] = "1"

    os.chdir(dex_root)
    if os.fspath(dex_root) not in sys.path:
        sys.path.insert(0, os.fspath(dex_root))

    _extend_sys_path_from_env(verbose=bool(args.verbose))
    _patch_simulation_app(args)
    _install_open3d_ml_stub(
        enabled=not bool(args.disable_open3d_ml_stub),
        verbose=bool(args.verbose),
    )

    forwarded_args = _clean_target_args(args.target_args)
    sys.argv = [os.fspath(target_path)] + forwarded_args
    if bool(args.verbose):
        print(
            "[env-wrapper] forwarding argv:",
            json.dumps(sys.argv, ensure_ascii=False),
            flush=True,
        )
        print(
            "[env-wrapper] gpu env:",
            json.dumps(
                {
                    "CUDA_VISIBLE_DEVICES": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                    "DEXGARMENTLAB_ACTIVE_GPU": str(os.environ.get("DEXGARMENTLAB_ACTIVE_GPU", "")),
                    "DEXGARMENTLAB_PHYSICS_GPU": str(os.environ.get("DEXGARMENTLAB_PHYSICS_GPU", "")),
                    "VK_ICD_FILENAMES": str(os.environ.get("VK_ICD_FILENAMES", "")),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if bool(args.run_as_main):
        runpy.run_path(os.fspath(target_path), run_name="__main__")
        return 0

    spec = importlib.util.spec_from_file_location("__dex_env_validation_target__", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {target_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if bool(args.call_main_if_present):
        target_main = getattr(module, "main", None)
        if callable(target_main):
            result = target_main()
            if isinstance(result, int):
                return int(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
