"""Runtime speed patches for the upstream dlss_engine (applied monkey-patch).

Both patches are applied at runtime so the vendored engine checkout stays
untouched and upstream updates cannot wipe them. They were A/B-measured in the
AutoTube pipeline (Sep 2026, 24 -> 60 fps on a real H3 scene clip) and cut wall
time ~4x at identical output. Defaults are on; pass ``zero_flow=False`` /
``min_cascade=False`` from the CLI to disable either.
"""

from __future__ import annotations

_flow_patch_installed = False
_cascade_patch_installed = False


def install_zero_flow_guide() -> None:
    """Skip the CPU DIS optical-flow guide (zero motion vectors).

    The upstream engine feeds CPU optical-flow vectors into the DLSSG worker as
    a guide. The DLSSG worker computes its own motion internally and ignores the
    CPU vectors, so the DIS optical-flow pass only costs time. A/B
    (couriers-008, Sep 2026): zero-flow vs engine flow produced identical frame
    counts (394 gen / 101 copy / 495 out) while cutting wall time 77s -> 30s.
    Only the cheap absdiff scene-cut/duplicate detection is kept; the DIS
    optical-flow pass is skipped entirely.
    """
    global _flow_patch_installed
    if _flow_patch_installed:
        return
    import cv2
    import numpy as np

    from dlss_engine.frame_interpolation import processor as proc_mod
    from dlss_engine.frame_interpolation.guides import (
        DLSSGGuideGenerator as BaseGuide,
        Guide,
    )

    class ZeroFlowGuide(BaseGuide):
        def process(self, rgba, *, force_reset: bool = False) -> Guide:
            current = self._gray(rgba)
            if self.previous is None:
                guide = Guide(self.zero, True, 1.0, False, 0.0)
            else:
                difference = cv2.absdiff(current, self.previous)
                score = float(np.mean(difference)) / 255.0
                duplicate = score < 0.0005
                reset = force_reset or score > 0.24
                guide = Guide(
                    self.zero,
                    reset,
                    score,
                    duplicate,
                    1.0 if duplicate else 0.0,
                )
            self.previous = current
            return guide

    proc_mod.DLSSGGuideGenerator = ZeroFlowGuide
    _flow_patch_installed = True


def install_min_stage_cascade() -> None:
    """Reduce Cascade stages to the minimum power-of-2 grid for the target rate.

    The engine scheduler hardcodes 3 stages for any non-2x/4x ratio, so 24 -> 60
    (2.5x) builds a 192 fps grid and makes the DLSSG worker do ~2.8x the calls
    needed. Only ceil(log2(2.5)) = 2 stages (96 fps grid) are required to sample
    60 fps. Output is identical (the extra stage only oversampled); combined
    with zero-flow this cuts 24 -> 60 wall time 77s -> ~18s (~4x).
    """
    global _cascade_patch_installed
    if _cascade_patch_installed:
        return
    from fractions import Fraction

    from dlss_engine.frame_interpolation import processor as proc_mod
    from dlss_engine.frame_interpolation.models import InterpolationPlan

    orig = proc_mod.choose_interpolation_plan

    def _plan(source_rate, target_rate, engine, native_max, *, cfr=True):
        plan = orig(source_rate, target_rate, engine, native_max, cfr=cfr)
        if plan.path != "Cascade":
            return plan
        ratio = target_rate / source_rate
        stages = 1
        while (1 << stages) * ratio.denominator < ratio.numerator:
            stages += 1
        grid = 1 << stages
        exact = cfr and ratio in {Fraction(2), Fraction(4)}
        return InterpolationPlan(
            path="Cascade",
            source_rate=source_rate,
            target_rate=target_rate,
            native_multiplier=2,
            grid_multiplier=grid,
            cascade_stages=stages,
            maximum_temporal_error=(
                Fraction(0) if exact else Fraction(1, 2 * grid) / source_rate
            ),
            generated_per_interval=grid - 1,
        )

    proc_mod.choose_interpolation_plan = _plan
    _cascade_patch_installed = True


def install(*, zero_flow: bool = True, min_cascade: bool = True) -> None:
    """Install the enabled patches. Call after the engine is importable."""
    if zero_flow:
        install_zero_flow_guide()
    if min_cascade:
        install_min_stage_cascade()
