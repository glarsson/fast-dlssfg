# fast-dlssfg

Standalone NVIDIA DLSS Frame Interpolation for video files: turn a 24 fps clip
into a buttery 60 fps clip with real generated intermediate frames, directly
from the command line. No ComfyUI server, no model downloads, one binary-free
Python package plus the NVIDIA engine it drives.

24FPS:
https://github.com/user-attachments/assets/b698c6da-f3be-43f2-8067-460cc4a3d010

60FPS:
https://github.com/user-attachments/assets/1c1cc1c8-036b-48bf-ad38-34f6e75244e9




```text
fast-dlssfg run input.mp4
-> input_60fps.mp4   (60 fps, DLSS-generated intermediate frames)
```

## Why this exists

The upstream [ComfyUI-NVIDIA-DLSS-Frame-Interpolation](https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation)
custom node wraps NVIDIA's DLSS Frame Generation runtime, but it is built for a
ComfyUI graph, and its defaults are slow. This repo extracts the frame-
interpolation engine into a plain CLI and ships two runtime patches that make
it **about 4x faster at identical output**:

1. **Zero-flow guide** - skip the CPU optical-flow guide. The upstream engine
   computes DIS optical-flow vectors on the CPU and feeds them to the DLSSG
   worker as a guide, but the DLSSG worker computes its own motion internally
   and ignores the CPU vectors. Keeping only the cheap absdiff scene-cut /
   duplicate detection costs nothing visually.
2. **Minimum-stage cascade** - the engine's scheduler hardcodes 3 cascade
   stages for any non-2x/4x ratio, so 24 -> 60 (2.5x) builds a 192 fps grid
   and makes the worker do ~2.8x the calls needed. Only 2 stages (a 96 fps
   grid) are required to sample 60 fps; the third stage only oversamples.

Both are applied as monkey-patches at runtime, so the engine checkout stays
untouched and upstream updates cannot wipe them. They can be disabled with
`--no-zero-flow` / `--full-cascade`.

### A/B evidence (Sep 2026, real 24 fps scene clip, RTX 5060 Ti)

| Variant | Generated / copied frames | Wall time |
| --- | ---: | ---: |
| Upstream engine defaults | 394 gen / 101 copy / 495 out | ~77 s |
| Zero-flow only | identical frame counts | ~30 s |
| Zero-flow + minimum-stage cascade | identical frame counts | ~18 s |

Same output frames in every variant; the difference is only how much the
engine oversamples and how much useless CPU optical flow it computes.

## Requirements

- Windows
- An NVIDIA RTX GPU
- A current NVIDIA display driver (Hardware-accelerated GPU scheduling /
  HAGS recommended for Frame Generation)
- FFmpeg and FFprobe on `PATH` (or set `DLSS_FFMPEG_PATH` / `DLSS_FFPROBE_PATH`
  to the executables)
- Python >= 3.9 with `pip` (dependencies: `av`, `numpy`, `opencv-python`)
- `git` with Git LFS, for the one-time engine provisioning

The engine needs no model weights and makes no network requests at runtime.

## Install

```bash
pip install -e .
```

## Provision the engine (once)

The NVIDIA DLSS runtime DLLs and the `dlssg-worker.exe` worker ship inside the
upstream custom-node checkout (tracked with Git LFS, ~230 MB). This repo never
vendors those binaries:

```bash
fast-dlssfg setup
```

This clones the upstream engine into `engine/` (gitignored) and materializes
its LFS objects. If you already have a checkout - for example the copy inside
a ComfyUI `custom_nodes/ComfyUI-NVIDIA-DLSS-Frame-Interpolation` folder - you
can skip the download entirely:

```bash
fast-dlssfg run input.mp4 --engine-dir D:/path/to/ComfyUI-NVIDIA-DLSS-Frame-Interpolation
# or once, for every future run:
set FAST_DLSSFG_ENGINE_DIR=D:/path/to/ComfyUI-NVIDIA-DLSS-Frame-Interpolation
```

The engine folder must contain materialized runtimes (real `nvngx_dlssg.dll`
is ~7 MB; a few-hundred-byte text file means `git lfs pull` never ran).

## Usage

```bash
# 24 -> 60 fps (the default target)
fast-dlssfg run input.mp4

# Several clips, explicit output dir
fast-dlssfg run a.mp4 b.mp4 c.mp4 --out-dir ./interpolated

# Full option set
fast-dlssfg run input.mp4 --fps 60 --engine Cascade --quality Max \
    --codec "H.264 (NVIDIA NVENC)" --out-dir out/
```

By default each output lands under `<repo>/out/<source-project>/<clip-stem>/`
as `<clip-stem>_60fps.mp4` (a `--out-dir` writes clips directly there).

### CLI reference

```
fast-dlssfg setup [--engine-dir DIR]
fast-dlssfg run CLIP [CLIP...] [options]

  --fps FPS          Target FPS (default 60; any rate up to 6x the source)
  --engine ENGINE    Auto | Native DLSSG | Cascade  (default Cascade)
  --quality QUALITY  Auto (Default) | Max | Best | Good  (default Max)
  --codec CODEC      H.264, H.265, AV1, ProRes Proxy + NVENC variants
                     (default "H.264 (NVIDIA NVENC)")
  --out-dir DIR      Override output directory
  --engine-dir DIR   Engine checkout location (overrides FAST_DLSSFG_ENGINE_DIR)
  --no-zero-flow     Keep the upstream CPU optical-flow guide (slower)
  --full-cascade     Use the upstream full cascade grid (slower, identical)
```

**Auto** uses native DLSSG when source and target form an exact 2x/4x rate,
and otherwise cascades. **Native DLSSG** accepts only exact multipliers
supported by the installed runtime. **Cascade** (default) works for any rate
and is what the speed patches optimize.

### Machine-readable progress

stdout is pure JSON, one object per line - easy to consume from a script:

```json
{"event": "start", "clip": "D:/in/input.mp4", "src": {"width": 960, "height": 544, "fps": 24.0, "frames": 211}, "fps": "60", "engine": "Cascade", "flow": "zero"}
{"event": "progress", "value": 0.4231, "desc": "Interpolating 7/9"}
{"event": "done", "clip": "D:/in/input.mp4", "engine_actual": "Cascade", "wall_clock_sec": 18.4, "engine_elapsed_sec": 18.2, "output": "D:/out/input_60fps.mp4", "out_probe": {"width": 960, "height": 544, "fps": 60.0, "frames": 495}, "frames_in": 211, "generated": 394, "copied": 101, "dup_fraction": 0.0, "report": "D:/out/input_60fps.report.json"}
```

Human progress and the end-of-run summary table go to stderr, so redirecting
`2>/dev/null` yields a clean event stream.

## How it works

The package drives `dlss_engine` (the Python library inside the upstream node)
directly - the same code the ComfyUI node runs, minus the Comfy queue and
process. Frame Generation works in temporal bursts: it renders new frames that
fit between existing ones, and the engine picks the output frames that land on
your exact target timeline. Scene cuts reset the interpolation history.

For 24 -> 60 the engine cascades 2x stages. A cascade of N stages produces a
`2^N` fps grid and samples the target rate from it. The engine hardcodes 3
stages for non-integer ratios, which for 24 -> 60 means a 192 fps grid even
though a 96 fps grid already contains every 60 fps timestamp exactly. The
minimum-stage patch computes the smallest power-of-two grid whose timestamps
cover the target rate (`ceil(log2(2.5)) = 2` stages), which yields identical
output with ~2.8x fewer DLSSG calls.

The DLSSG worker is launched per run; each output file is encoded once
(NVENC by default). If the engine's encode drops the source audio track, the
audio is re-muxed onto the result (AAC 192k) so an interpolated file is never
silent. Failures remove the partial output - you never get a half-written file
at the final name.

## Validation

Each finished file is probed and a duplicate-like fraction is measured: the
first ~240 frames are decoded to gray and adjacent pairs count as duplicated
when <= 5% of pixels changed by more than 2/255 (lossy NVENC re-encode makes
exact frame equality useless). A naive 24 -> 60 "frame doubling" reference
measures ~0.6; DLSS-generated output measures near 0.0, meaning almost every
output frame is genuinely new. The numbers land in the `done` event and the
`.report.json` beside each output (engine JSON diagnostics: negotiated plan,
generated/copied frames, scene cuts).

## Troubleshooting

**"No usable DLSS engine at ..."** - run `fast-dlssfg setup`, or point
`FAST_DLSSFG_ENGINE_DIR` / `--engine-dir` at an existing checkout.

**Runtime DLL is missing or only a few bytes** - the Git LFS objects were not
downloaded. Run `git lfs pull` inside the engine folder, or re-run
`fast-dlssfg setup`. Real `nvngx_dlssg.dll` is ~7 MB.

**ffprobe / ffmpeg not found** - install FFmpeg and add it to `PATH`, or set
`DLSS_FFPROBE_PATH` / `DLSS_FFMPEG_PATH`.

**Frame Generation is unavailable** - update the NVIDIA driver, enable HAGS in
Windows graphics settings, restart, and retry. The engine report includes the
detected GPU, driver, and supported native multiplier.

**Native DLSSG rejects the requested FPS** - use Cascade (the default) or Auto.

## Licensing

- This repo (the CLI wrapper, patches, docs): **MIT**, see `LICENSE`.
- The engine is the upstream [Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation](https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation)
  project, derived from [Merserk/dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer)
  (MIT). Its license texts live inside the checkout.
- The bundled NVIDIA DLSS runtimes carry their own license, which includes a
  **commercial-release notification requirement** for applications that
  incorporate the DLSS SDK. Review `engine/bin/runtime/*/LICENSE-NVIDIA-DLSS.txt`
  before redistribution or commercial use.

fast-dlssfg does not include the NVIDIA runtime files in its own repository;
`fast-dlssfg setup` fetches them into a gitignored directory.

## Out of scope

This is the fast frame-interpolation CLI only. The DLSS video upscale and image
upscale paths of the upstream node, and any ComfyUI-graph integration, are not
covered here - but the same `dlss_engine` checkout drives all three, so the
setup step above works for them too.
