from __future__ import annotations

import importlib
import os
import pathlib
import shutil
import sys
import inspect
from typing import Iterable

REQUIRED_TRANSFORMERS_VERSION = "4.53.2"
_RUNTIME_PYDEPS_MARKERS = (
    "WorldModelDiffusionVlaRuntime/pydeps",
    "WorldModelDiffusionVlaRuntime\\pydeps",
)
_KEEP_RUNTIME_PYDEPS_ENV = "MYVLA_KEEP_RUNTIME_PYDEPS"


def _get_package_root() -> pathlib.Path:
    # myVLA/myvla_pi05/transformers_patch.py -> myVLA
    return pathlib.Path(__file__).resolve().parents[1]


def _get_patch_dir() -> pathlib.Path:
    return _get_package_root() / "vendor" / "transformers_replace"


def _get_transformers_root() -> pathlib.Path:
    import transformers

    return pathlib.Path(transformers.__file__).resolve().parent


def _is_transformers_replace_installed() -> bool:
    try:
        mod = importlib.import_module("transformers.models.siglip.check")
    except Exception:  # noqa: BLE001
        return False
    fn = getattr(mod, "check_whether_transformers_replace_is_installed_correctly", None)
    if callable(fn):
        try:
            if not bool(fn()):
                return False
        except Exception:  # noqa: BLE001
            return False
    else:
        return False
    # Ensure the installed patch includes MEM video-encoder support (5D pixel_values path).
    try:
        modeling = importlib.import_module("transformers.models.siglip.modeling_siglip")
        vision_transformer = getattr(modeling, "SiglipVisionTransformer", None)
        if vision_transformer is None or not hasattr(vision_transformer, "_forward_video"):
            return False
    except Exception:  # noqa: BLE001
        return False

    # Ensure the Gemma patch is present (AdaRMS cond_dim support) since pi0.5 expert uses it.
    try:
        gemma = importlib.import_module("transformers.models.gemma.modeling_gemma")
        rms_norm = getattr(gemma, "GemmaRMSNorm", None)
        if rms_norm is None or "cond_dim" not in inspect.signature(rms_norm.__init__).parameters:
            return False
    except Exception:  # noqa: BLE001
        return False

    return True


def _copy_into_dir(*, src_items: Iterable[pathlib.Path], dst_dir: pathlib.Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_items:
        dst = dst_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _sanitize_transformers_import_env(*, verbose: bool) -> None:
    if str(os.environ.get(_KEEP_RUNTIME_PYDEPS_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        if verbose:
            print(f"Keeping runtime pydeps overlay because {_KEEP_RUNTIME_PYDEPS_ENV}=1")
        return

    removed_sys_paths: list[str] = []
    kept_sys_path: list[str] = []
    for item in sys.path:
        text = os.fspath(item)
        if any(marker in text for marker in _RUNTIME_PYDEPS_MARKERS):
            removed_sys_paths.append(text)
            continue
        kept_sys_path.append(item)
    if removed_sys_paths:
        sys.path[:] = kept_sys_path

    raw_pythonpath = str(os.environ.get("PYTHONPATH", "") or "")
    had_runtime_overlay = bool(removed_sys_paths)
    if raw_pythonpath:
        pythonpath_tokens = [token for token in raw_pythonpath.split(os.pathsep) if token]
        kept_pythonpath = [
            token
            for token in pythonpath_tokens
            if not any(marker in token for marker in _RUNTIME_PYDEPS_MARKERS)
        ]
        if len(kept_pythonpath) != len(pythonpath_tokens):
            had_runtime_overlay = True
        if kept_pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept_pythonpath)
        else:
            os.environ.pop("PYTHONPATH", None)

    if had_runtime_overlay:
        for name in tuple(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                sys.modules.pop(name, None)
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                sys.modules.pop(name, None)
            if name == "numpy" or name.startswith("numpy."):
                sys.modules.pop(name, None)
            if name == "scipy" or name.startswith("scipy."):
                sys.modules.pop(name, None)
            if name == "sklearn" or name.startswith("sklearn."):
                sys.modules.pop(name, None)
        for key in tuple(sys.path_importer_cache):
            text = os.fspath(key)
            if any(marker in text for marker in _RUNTIME_PYDEPS_MARKERS):
                sys.path_importer_cache.pop(key, None)
        importlib.invalidate_caches()
        if verbose:
            print(
                "Sanitized model import path by removing runtime pydeps overlay: "
                + ", ".join(removed_sys_paths or ["<pythonpath-only>"])
            )


def ensure_transformers_replace_installed(*, verbose: bool = True) -> None:
    """Ensure the transformers_replace patch is applied for PI0/PI0.5 PyTorch inference.

    openpi's pi0/pi05 PyTorch model expects a patched Transformers layout where
    `transformers.models.siglip.check` exists and validates the Transformers version.
    """
    # This project is PyTorch-only. If TensorFlow is installed in the environment but
    # incompatible with the current protobuf version, importing Transformers can spam
    # errors or even fail. Prefer opting out of TF/Flax before importing Transformers.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    _sanitize_transformers_import_env(verbose=bool(verbose))

    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Unsupported transformers version: {transformers.__version__}. "
            f"Expected {REQUIRED_TRANSFORMERS_VERSION}."
        )

    if _is_transformers_replace_installed():
        return

    patch_dir = _get_patch_dir()
    if not patch_dir.is_dir():
        raise FileNotFoundError(
            f"transformers_replace patch directory not found: {patch_dir}. "
            "Make sure myVLA/vendor/transformers_replace exists."
        )

    transformers_root = _get_transformers_root()
    if verbose:
        print(f"Applying transformers_replace patch into: {transformers_root}")

    _copy_into_dir(src_items=patch_dir.iterdir(), dst_dir=transformers_root)

    # The current Python process may have imported old `transformers.models.*` modules already.
    # Invalidate caches and force a re-import so our post-copy check sees the updated files.
    importlib.invalidate_caches()
    for name in (
        "transformers.models.siglip",
        "transformers.models.siglip.check",
        "transformers.models.siglip.modeling_siglip",
        "transformers.models.gemma",
        "transformers.models.gemma.configuration_gemma",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.paligemma",
        "transformers.models.paligemma.modeling_paligemma",
    ):
        sys.modules.pop(name, None)

    if not _is_transformers_replace_installed():
        raise RuntimeError(
            "transformers_replace patch copy finished, but the installed Transformers files still "
            "do not match what pi0/pi0.5 expects. Check file permissions and your Python environment."
        )


def maybe_disable_torch_compile_from_env() -> bool:
    """Return True if torch.compile should be disabled (for faster first-run / compatibility)."""
    return os.environ.get("MYVLA_DISABLE_TORCH_COMPILE", "").strip() not in ("", "0", "false", "False")
