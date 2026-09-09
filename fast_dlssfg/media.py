"""FFmpeg/ffprobe helpers: media probe, audio detection, audio backstop.

The interpolation engine encodes a new video stream; whether it carries the
source audio depends on the codec/container path. fast-dlssfg guarantees a
non-silent output: if the interpolated master dropped the audio, the source
audio track is re-muxed onto it (AAC 192k). Later stages pull audio from this
file, so a silent master would silently kill delivery audio.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def _ffprobe_bin() -> str:
    found = os.environ.get("DLSS_FFPROBE_PATH") or shutil.which("ffprobe")
    if not found:
        raise RuntimeError(
            "ffprobe not found on PATH. Install FFmpeg, or set DLSS_FFPROBE_PATH "
            "to the ffprobe.exe location."
        )
    return found


def _ffmpeg_bin() -> str:
    found = os.environ.get("DLSS_FFMPEG_PATH") or shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install FFmpeg, or set DLSS_FFMPEG_PATH "
            "to the ffmpeg.exe location."
        )
    return found


def probe(path: Path) -> dict[str, object]:
    """Return width, height, fps (numeric), and frames via ffprobe."""
    path = Path(path)
    cmd = [
        _ffprobe_bin(),
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    data = json.loads(out) or {}
    stream = (data.get("streams") or [{}])[0]
    rate = str(stream.get("avg_frame_rate") or "0/1")
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if den else 0.0
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": round(fps, 4),
        "frames": int(stream.get("nb_read_frames") or 0),
    }


def has_audio(path: Path) -> bool:
    """True when the file carries at least one audio stream."""
    path = Path(path)
    result = subprocess.run(
        [
            _ffprobe_bin(),
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return "audio" in (result.stdout or "").lower()


def backstop_audio(src: Path, produced: Path) -> bool:
    """Re-mux source audio onto ``produced`` if the engine dropped it.

    No-op when the source has no audio or the master already does. Writes a
    sibling temp and atomically replaces ``produced`` on success. Returns True
    when the final file is known to carry audio.
    """
    src = Path(src)
    produced = Path(produced)
    try:
        if not has_audio(src) or has_audio(produced):
            return has_audio(produced)
    except RuntimeError:
        return False
    remux = produced.with_name(f"_fg_audio_{src.stem}.mp4")
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(produced),
        "-i",
        str(src),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(remux),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        remux.unlink(missing_ok=True)
        return False
    if remux.is_file():
        remux.replace(produced)
        return True
    return False


def dup_fraction(path: Path, max_frames: int = 240) -> dict[str, float]:
    """Duplicate-like adjacent-frame fraction via PyAV + numpy (lossy-safe).

    Lossy re-encode (NVENC) makes exact frame equality useless for detecting
    duplicated frames, so this decodes gray and counts adjacent pairs where <=5%
    of pixels changed by more than 2/255 (a dup-up 24 -> 60 reference measures
    ~0.6). Bounded to the first ``max_frames`` frames so the QA pass is O(1) per
    clip instead of a full re-decode. QA-only; delivery frames are untouched.
    """
    import av
    import numpy as np

    try:
        container = av.open(str(path))
        prev = None
        pairs = 0
        dup_like = 0
        madiffs = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="gray")
            if prev is not None:
                diff = np.abs(arr.astype(np.int16) - prev)
                madiffs.append(float(diff.mean()))
                if float((diff > 2).mean()) <= 0.05:
                    dup_like += 1
                pairs += 1
                if pairs >= max_frames:
                    break
            prev = arr
        container.close()
    except Exception:
        return {"dup_fraction": -1.0, "mean_abs_diff": -1.0}
    if pairs == 0:
        return {"dup_fraction": 0.0, "mean_abs_diff": 0.0}
    return {
        "dup_fraction": round(dup_like / pairs, 4),
        "mean_abs_diff": round(sum(madiffs) / len(madiffs), 4),
    }
