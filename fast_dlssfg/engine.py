"""Engine discovery and provisioning for fast-dlssfg.

The actual NVIDIA DLSS FG engine is the upstream ComfyUI custom node
``Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation``. It ships the bundled
NVIDIA DLSS Frame Generation runtimes (``dlssg-worker.exe``, ``nvngx_dlssg.dll``
and friends, tracked with Git LFS) and the ``dlss_engine`` Python package that
drives them. ``dlss_engine.core.paths`` resolves every runtime path relative to
the node folder (``ROOT = parents[2]``), so the engine must live on disk as a
folder that is added to ``sys.path`` - it cannot be pip-installed.

fast-dlssfg never vendors these binaries. ``fast-dlssfg setup`` clones the
upstream node into ``engine/`` (gitignored) and materializes its LFS objects;
``FAST_DLSSFG_ENGINE_DIR`` or ``--engine-dir`` can point at an existing checkout
instead (e.g. the one inside a ComfyUI ``custom_nodes/`` install).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_REPO = "https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation.git"
NODE_FOLDER_NAME = "ComfyUI-NVIDIA-DLSS-Frame-Interpolation"
ENGINE_ENV = "FAST_DLSSFG_ENGINE_DIR"

# Real materialized LFS runtime DLLs are multi-megabyte; an LFS *pointer* file
# is a few hundred bytes of text. Size gate catches a clone where `git lfs
# pull` was never run.
_MIN_RUNTIME_BYTES = 1_000_000


class EngineError(RuntimeError):
    pass


def repo_root() -> Path:
    """Top-level fast-dlssfg checkout."""
    return Path(__file__).resolve().parents[1]


def default_engine_dir() -> Path:
    """Repo-relative default: <repo>/engine/ComfyUI-NVIDIA-DLSS-Frame-Interpolation."""
    return repo_root() / "engine" / NODE_FOLDER_NAME


def discover_engine_dir(explicit: Path | str | None = None) -> Path:
    """Resolve the engine folder: --engine-dir > FAST_DLSSFG_ENGINE_DIR > default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ENGINE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return default_engine_dir()


def _check_git_lfs() -> None:
    if shutil.which("git") is None:
        raise EngineError("git not found on PATH; the engine is installed from a git clone.")
    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        # git-lfs may be a git subcommand without its own shim; probe anyway.
        probe = subprocess.run(
            ["git", "lfs", "version"], capture_output=True, text=True
        )
        if probe.returncode != 0:
            raise EngineError(
                "git-lfs is required to materialize the NVIDIA runtime DLLs "
                "(git lfs version failed). Install Git LFS, then re-run "
                "'fast-dlssfg setup'."
            )


def verify_engine(engine_dir: Path) -> bool:
    """True when the checkout exists and the DLSSG runtime is materialized."""
    if not engine_dir.is_dir():
        return False
    worker = engine_dir / "bin" / "runtime" / "dlssg" / "dlssg-worker.exe"
    runtime = engine_dir / "bin" / "runtime" / "dlssg" / "nvngx_dlssg.dll"
    pkg = engine_dir / "dlss_engine" / "frame_interpolation" / "processor.py"
    if not (worker.is_file() and pkg.is_file()):
        return False
    if not runtime.is_file():
        return False
    try:
        if runtime.stat().st_size < _MIN_RUNTIME_BYTES:
            return False  # LFS pointer stub, not the real DLL
    except OSError:
        return False
    return True


def install_engine(dest: Path, *, quiet: bool = False) -> Path:
    """Clone the upstream node into ``dest`` and materialize its LFS runtimes.

    Returns the resolved engine dir. Raises EngineError on any step that
    fails. Never touches an existing checkout.
    """
    _check_git_lfs()
    dest = dest.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        if verify_engine(dest):
            return dest
        raise EngineError(
            f"{dest} exists but is not a usable engine checkout. Remove it and "
            "re-run 'fast-dlssfg setup', or point FAST_DLSSFG_ENGINE_DIR at a "
            "complete ComfyUI-NVIDIA-DLSS-Frame-Interpolation folder."
        )
    dest.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[fast-dlssfg] cloning {UPSTREAM_REPO}")
        print(f"[fast-dlssfg]   -> {dest}")
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(dest)],
        text=True,
    )
    if clone.returncode != 0:
        raise EngineError("git clone of the DLSS engine failed; see output above.")
    if not quiet:
        print("[fast-dlssfg] pulling git-lfs runtime objects (~230 MB)...")
    lfs = subprocess.run(["git", "lfs", "pull"], cwd=dest, text=True)
    if lfs.returncode != 0:
        raise EngineError("git lfs pull failed; the runtime DLLs are pointer stubs.")
    if not verify_engine(dest):
        raise EngineError(
            "engine cloned but the DLSSG runtime did not materialize. Check "
            "that git-lfs is installed and the clone contains real DLLs."
        )
    if not quiet:
        print("[fast-dlssfg] engine ready.")
    return dest


def ensure_engine(explicit: Path | str | None = None, *, auto_install: bool = False) -> Path:
    """Resolve + verify the engine; optionally provision it when missing."""
    engine_dir = discover_engine_dir(explicit)
    if verify_engine(engine_dir):
        return engine_dir
    if auto_install:
        return install_engine(engine_dir)
    raise EngineError(
        f"No usable DLSS engine at {engine_dir}.\n"
        "Run 'fast-dlssfg setup' to clone it, or set "
        f"{ENGINE_ENV} (or pass --engine-dir) to an existing "
        "ComfyUI-NVIDIA-DLSS-Frame-Interpolation folder."
    )


def inject_engine(engine_dir: Path) -> None:
    """Add the engine folder to sys.path so `import dlss_engine` resolves.

    Idempotent; safe to call before every run.
    """
    engine_dir = engine_dir.expanduser().resolve()
    node = engine_dir if engine_dir.name == NODE_FOLDER_NAME else engine_dir
    if str(node) not in sys.path:
        sys.path.insert(0, str(node))
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))


def probe_engine_imports(engine_dir: Path) -> None:
    """Import-check the pieces fast-dlssfg needs from the engine checkout."""
    inject_engine(engine_dir)
    try:
        import dlss_engine  # noqa: F401
        from dlss_engine.frame_interpolation.models import (  # noqa: F401
            FrameInterpolationOptions,
        )
        from dlss_engine.frame_interpolation.processor import (  # noqa: F401
            interpolate_video,
        )
    except Exception as exc:  # noqa: BLE001
        raise EngineError(
            f"dlss_engine import failed from {engine_dir}: {exc}"
        ) from exc
