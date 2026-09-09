"""fast-dlssfg command line: setup and run.

    fast-dlssfg setup [--engine-dir DIR]          provision the DLSS engine
    fast-dlssfg run CLIP [CLIP...] [options]      interpolate clip(s) to 60 fps

stdout is machine-readable: line-delimited JSON events (start / progress /
done). Human progress + the summary table go to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .engine import (
    EngineError,
    default_engine_dir,
    discover_engine_dir,
    ensure_engine,
    install_engine,
    probe_engine_imports,
)
from .interpolate import print_summary, run_one

ENGINE_CHOICES = ("Auto", "Native DLSSG", "Cascade")
QUALITY_CHOICES = ("Auto (Default)", "Max", "Best", "Good")
CODEC_CHOICES = (
    "H.264",
    "H.264 (NVIDIA NVENC)",
    "H.265",
    "H.265 (NVIDIA NVENC)",
    "AV1",
    "AV1 (NVIDIA NVENC)",
    "ProRes Proxy",
)


def _resolve_dest(clip: Path, out_dir: Path | None) -> Path:
    """Output directory for one clip.

    With --out-dir the clip writes directly there; by default outputs land
    under <repo>/out/<parent-project>/<clip-stem>/.
    """
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()
    from .engine import repo_root

    return repo_root() / "out" / clip.parent.parent.name / clip.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-dlssfg",
        description=(
            "Standalone NVIDIA DLSS Frame Interpolation (24 fps clips -> 60 fps). "
            "Drives the dlss_engine package directly; no ComfyUI needed. Run "
            "'fast-dlssfg setup' once to provision the engine."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Clone + provision the DLSS engine (~230 MB).")
    setup.add_argument(
        "--engine-dir",
        default=None,
        help=f"Where to install the engine (default: {default_engine_dir()}).",
    )

    run = sub.add_parser(
        "run", help="Interpolate one or more clips to the target fps."
    )
    run.add_argument("clips", nargs="+", help="Input video file(s).")
    run.add_argument("--fps", default="60", help="Target FPS (60 recommended).")
    run.add_argument(
        "--engine",
        default="Cascade",
        choices=ENGINE_CHOICES,
        help="Cascade recommended (native DLSSG accepts only exact 2x/4x).",
    )
    run.add_argument("--quality", default="Max", choices=QUALITY_CHOICES)
    run.add_argument(
        "--codec",
        default="H.264 (NVIDIA NVENC)",
        choices=CODEC_CHOICES,
        help="NVENC variants encode on the GPU.",
    )
    run.add_argument("--out-dir", default=None, help="Override output directory.")
    run.add_argument(
        "--engine-dir",
        default=None,
        help="Path to a ComfyUI-NVIDIA-DLSS-Frame-Interpolation checkout "
        "(default: FAST_DLSSFG_ENGINE_DIR, then <repo>/engine/).",
    )
    run.add_argument(
        "--no-zero-flow",
        action="store_true",
        help="Keep the upstream CPU optical-flow guide (slower, identical output).",
    )
    run.add_argument(
        "--full-cascade",
        action="store_true",
        help="Use the upstream full Cascade grid (192 fps for 24->60; ~2.8x "
        "more DLSSG calls, identical output).",
    )
    return parser


def cmd_setup(args: argparse.Namespace) -> int:
    dest = discover_engine_dir(args.engine_dir)
    install_engine(dest)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    clips = [Path(c).expanduser().resolve() for c in args.clips]
    missing = [str(c) for c in clips if not c.is_file()]
    if missing:
        print(f"[fast-dlssfg] clip(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    engine_dir = ensure_engine(args.engine_dir)
    probe_engine_imports(engine_dir)

    from . import patches

    patches.install(zero_flow=not args.no_zero_flow, min_cascade=not args.full_cascade)

    summary = []
    for clip in clips:
        dest = _resolve_dest(clip, Path(args.out_dir) if args.out_dir else None)
        try:
            summary.append(
                run_one(
                    clip,
                    output_fps=args.fps,
                    engine=args.engine,
                    quality=args.quality,
                    codec=args.codec,
                    dest_dir=dest,
                )
            )
        except Exception as exc:  # noqa: BLE001 - clean per-run failure, no traceback
            print(f"[fast-dlssfg] error interpolating {clip.name}: {exc}", file=sys.stderr)
            return 1
    print_summary(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "setup":
            return cmd_setup(args)
        if args.command == "run":
            return cmd_run(args)
    except EngineError as exc:
        print(f"[fast-dlssfg] error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
