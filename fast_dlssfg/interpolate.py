"""Core 24 -> 60 (or any 6x-or-under) DLSS frame-interpolation driver.

Mirrors the AutoTube dlssfg worker: drives ``dlss_engine.frame_interpolation.
interpolate_video`` directly (no ComfyUI server), streams line-delimited JSON
events to stdout for scripting, and runs the two speed patches (see patches.py)
before the first clip.

JSON events:
  {"event": "start", "clip": "...", "src": {...}, "fps": "60", "engine": "Cascade", "flow": "zero"}
  {"event": "progress", "value": 0.0-1.0, "desc": "..."}
  {"event": "done", "clip": "...", "engine_actual": "...", "wall_clock_sec": 1.2,
   "engine_elapsed_sec": 1.1, "output": "...", "out_probe": {...},
   "frames_in": N, "generated": N, "copied": N, "dup_fraction": 0.0, "report": "..."}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .media import backstop_audio, dup_fraction, probe


def _log(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def _ascii_safe(text: str) -> str:
    """Engine labels may carry non-ASCII glyphs (e.g. an arrow U+2192 in the
    plan line); the console path here is ASCII-only, so replace them."""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)


def run_one(
    clip: Path,
    *,
    output_fps: str,
    engine: str,
    quality: str,
    codec: str,
    dest_dir: Path,
) -> dict[str, Any]:
    """Interpolate one clip to ``output_fps`` fps and write ``<stem>_<fps>fps.mp4``.

    Imports the engine lazily so the CLI can print usage errors without the
    engine present. Raises SystemExit on a missing clip; propagates engine
    failures with the interpolated output left unrenamed (never a partial
    output at the final name). Returns the ``done`` payload.
    """
    from dlss_engine.frame_interpolation.models import FrameInterpolationOptions
    from dlss_engine.frame_interpolation.processor import interpolate_video

    if not clip.is_file():
        raise SystemExit(f"[fast-dlssfg] clip not found: {clip}")
    dest = dest_dir.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    probe_src = probe(clip)
    _log(
        {
            "event": "start",
            "clip": str(clip),
            "src": probe_src,
            "fps": output_fps,
            "engine": engine,
            "flow": "zero",
        }
    )

    suffix = f"_{output_fps}fps"
    produced = dest / f"{clip.stem}{suffix}.mp4"
    produced.unlink(missing_ok=True)  # engine custom-rename refuses collisions
    opts = FrameInterpolationOptions(
        target_fps=output_fps,
        engine=engine,
        codec=codec,
        container="MP4",
        quality=quality,
        hdr_mode=False,
        rename_mode="Custom",
        custom_suffix=suffix,
    )
    started = time.time()

    _progress_bucket = [-1]  # human console log throttled to ~5% steps

    def _on_progress(value: float, desc: str) -> None:
        _log(
            {
                "event": "progress",
                "value": round(max(0.0, min(1.0, float(value))), 4),
                "desc": str(desc),
            }
        )
        bucket = min(20, int(float(value) * 20.0))
        if bucket != _progress_bucket[0]:
            _progress_bucket[0] = bucket
            print(
                f"[fast-dlssfg] {round(float(value) * 100.0):>3}% "
                f"{_ascii_safe(str(desc))}",
                file=sys.stderr,
                flush=True,
            )

    result = interpolate_video(
        str(clip),
        opts,
        progress=_on_progress,
        output_directory=str(dest),
        jobs_directory=str(dest / "jobs"),
        logs_directory=str(dest),
    )
    wall_clock_sec = round(time.time() - started, 1)
    output_path = Path(result.output_path)
    if not output_path.is_file():
        raise RuntimeError(
            f"engine finished but produced no file at {output_path} "
            f"(report: {getattr(result, 'report_path', '?')})"
        )
    # Guarantee a non-silent master (re-mux source audio when the engine
    # dropped it; -c:v copy keeps the video stream untouched).
    backstop_audio(clip, output_path)
    probe_out = probe(output_path)
    dup = dup_fraction(output_path)

    report_dest: Path | None = None
    src_report = Path(getattr(result, "report_path", "") or "")
    if src_report.exists():
        report_dest = dest / f"{clip.stem}{suffix}.report.json"
        report_dest.write_text(src_report.read_text(encoding="utf-8"), encoding="utf-8")

    payload: dict[str, Any] = {
        "event": "done",
        "clip": str(clip),
        "engine_actual": getattr(result, "selected_path", None) or engine,
        "wall_clock_sec": wall_clock_sec,
        "engine_elapsed_sec": round(getattr(result, "elapsed_seconds", 0.0), 1),
        "output": str(output_path.resolve()),
        "out_probe": probe_out,
        "frames_in": probe_src.get("frames"),
        "generated": getattr(result, "generated_frames", 0),
        "copied": getattr(result, "copied_frames", 0),
        "dup_fraction": dup.get("dup_fraction"),
    }
    if report_dest is not None:
        payload["report"] = str(report_dest.resolve())
    _log(payload)
    # Engine removes its per-run temp on success; only an empty dir remains.
    jobs_dir = dest / "jobs"
    try:
        jobs_dir.rmdir()
    except OSError:
        pass
    # Engine/upstream may also drop its own per-run report JSONs in dest.
    for leftover in dest.glob(f"DLSSFG_{clip.stem}_*.report.json"):
        leftover.unlink(missing_ok=True)
    return payload


def print_summary(results: list[dict[str, Any]]) -> None:
    """Human-readable summary table after a batch run (stderr - stdout stays JSON)."""
    out = sys.stderr
    print("\n[fast-dlssfg] summary", file=out)
    print(
        f"{'clip':52} {'eng_used':13} {'wall_s':>7} {'gen':>5} {'out_fr':>6} "
        f"{'dup':>7}",
        file=out,
    )
    for s in results:
        clip = Path(s["clip"]).parent.parent.name + "/" + Path(s["clip"]).stem
        print(
            f"{clip:52} {s['engine_actual']:13} {s['wall_clock_sec']:>7} "
            f"{s['generated']:>5} {(s.get('out_probe') or {}).get('frames', 0):>6} "
            f"{s['dup_fraction']:>7}",
            file=out,
        )
