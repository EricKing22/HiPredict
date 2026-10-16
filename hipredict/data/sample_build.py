"""Layer 1: shared, behaviour-preserving primitives for turning raw HiRISE bands
into model inputs.

This module is the single source of truth for "Job A" -- normalization statistics
and band/neighbour channel construction -- so the training dataset and the inference
engine can no longer drift apart.

It deliberately contains two *distinct* neighbour assemblers, because the two
workflows build neighbour context differently and both are load-bearing:

* Patch/context mode (training + inference patch loading) stacks the first three
  channels of the pre-computed RED5/RED3 context patches:
  ``[R5_c0, R5_c1, R5_c2, R3_c0, R3_c1, R3_c2]``.
* Strip/reconstruct mode synthesises neighbour context from the 32-column edges of
  the full-strip bands, in a different channel order:
  ``[right_bg, right_ir, left_r5, left_bg, left_ir, right_r3]``.

Normalization is exposed as a small pluggable ``Normalizer`` strategy. Downstream
users who need a bespoke normalization can pass their own ``Normalizer`` to the
inference engine (and, from Phase 2, to the datasets) without touching this module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from hipredict.data.normalization import (
    MAD_TO_NORMAL_STD,
    _robust_center_scale_from_predictors_np,
)

# Default neighbour-context width (columns taken from each strip edge).
CONTEXT_WIDTH = 32


# --------------------------------------------------------------------------- #
# Robust normalization statistics
# --------------------------------------------------------------------------- #
def _robust_center_scale_from_inputs(bands, neighbours=None, include_red_means=False, eps=1e-8):
    # bands: (2,H,W) -> IR, BG
    ir = bands[0].reshape(-1)
    bg = bands[1].reshape(-1)
    flat = torch.cat([ir, bg], dim=0)  # only pixel-aligned fields

    if include_red_means and neighbours is not None:
        # R5/R3 exist but are not pixel-aligned; use their per-patch means as *scalars*
        # neighbours shape ~ (2, H, Wn, 4), with RED at [..., 2]
        r5_mean = neighbours[0, :, :, 2].mean().reshape(1).to(flat.dtype)
        r3_mean = neighbours[1, :, :, 2].mean().reshape(1).to(flat.dtype)
        flat = torch.cat([flat, r3_mean, r5_mean], dim=0)

    a = flat.median()
    mad = (flat - a).abs().median()
    s = (MAD_TO_NORMAL_STD * mad).clamp_min(1e-3)
    return a, s


def _smooth_1d(x, kernel_size):
    x = x.reshape(1, 1, -1)
    n = int(x.shape[-1])
    if kernel_size <= 1 or n <= 1:
        return x.reshape(-1)

    k = min(int(kernel_size), n if n % 2 == 1 else max(1, n - 1))
    if k <= 1:
        return x.reshape(-1)

    pad = k // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    return F.avg_pool1d(x_pad, kernel_size=k, stride=1).reshape(-1)


def smooth_center_scale_from_inputs(bands, kernel_size):
    """Build a smooth row-wise center field from IR/BG while keeping one
    strip-global robust scale.

    This preserves large-scale brightness continuity better than a fully
    row-wise affine while still allowing a slowly varying offset a(y).
    """
    if bands.ndim != 3:
        raise ValueError(f"Expected bands with shape (2, H, W), got {tuple(bands.shape)}")

    ir = bands[0]
    bg = bands[1]
    h = int(ir.shape[0])

    flat_rows = torch.cat(
        [ir.reshape(h, -1), bg.reshape(h, -1)],
        dim=1,
    )

    row_center = flat_rows.median(dim=1).values

    a = _smooth_1d(row_center, kernel_size).to(dtype=bands.dtype)

    # One robust scene-wide scale to avoid introducing row-dependent
    # contrast changes across continuous landforms.
    flat_scene = flat_rows.reshape(-1)
    scene_center = flat_scene.median()
    scene_mad = (flat_scene - scene_center).abs().median()
    s = (MAD_TO_NORMAL_STD * scene_mad).clamp_min(1e-3).to(dtype=bands.dtype)

    return a.view(-1, 1), s


# --------------------------------------------------------------------------- #
# Pluggable normalizer strategies
# --------------------------------------------------------------------------- #
class Normalizer:
    """Strategy that maps (bands, neighbours) -> (center a, scale s).

    ``a``/``s`` may be scalars or broadcastable tensors; callers apply the affine
    ``(x - a) / s`` themselves so that the same stats can also be applied to a label
    (training) and recorded for later denormalization (inference).
    """

    def stats(self, bands, neighbours=None):
        raise NotImplementedError

    def predictor_stats(self, bg, ir):
        """Optional observation-level stats from full predictor arrays, computed
        before tiling in strip inference.

        Return ``None`` (the default) to signal that the strip path should not
        precompute stats and should instead normalize each tile via :meth:`stats`.
        """
        return None


class SceneRobust(Normalizer):
    """Per-scene robust (median / MAD) statistics from the pixel-aligned IR/BG."""

    def __init__(self, include_red_means=False):
        self.include_red_means = include_red_means

    def stats(self, bands, neighbours=None):
        return _robust_center_scale_from_inputs(
            bands, neighbours, include_red_means=self.include_red_means
        )

    def predictor_stats(self, bg, ir):
        # Canonical observation-level IR/BG stats used for full-strip inference.
        return _robust_center_scale_from_predictors_np(bg, ir)


class SceneSmooth(Normalizer):
    """Row-wise smooth center field with a single strip-global robust scale."""

    def __init__(self, kernel_size):
        self.kernel_size = int(kernel_size)

    def stats(self, bands, neighbours=None):
        return smooth_center_scale_from_inputs(bands, self.kernel_size)


class ConstantNormalizer(Normalizer):
    """Fixed (a, s) returned as-is -- used for global_robust and the identity case.

    The values are returned unchanged (not coerced), so callers keep exactly the
    scalar types they relied on previously (Python float for global_robust, int for
    the identity/None case).
    """

    def __init__(self, center, scale):
        self.center = center
        self.scale = scale

    def stats(self, bands=None, neighbours=None):
        return self.center, self.scale


def normalizer_from_mode(norm_mode, *, global_center=None, global_scale=None,
                         smooth_kernel=None, include_red_means=False):
    """Construct the built-in normalizer matching a legacy ``norm_mode`` string.

    Preserves the exact (a, s) types the previous inline logic produced:
    ``global_robust`` -> Python floats, ``None`` -> ints (0, 1).
    """
    if norm_mode == "scene_robust":
        return SceneRobust(include_red_means=include_red_means)
    if norm_mode == "scene_smooth":
        if smooth_kernel is None:
            raise ValueError("smooth_kernel is required for scene_smooth normalization")
        return SceneSmooth(smooth_kernel)
    if norm_mode == "global_robust":
        if global_center is None or global_scale is None:
            raise ValueError("global_center/global_scale required for global_robust")
        return ConstantNormalizer(float(global_center), float(global_scale))
    if norm_mode is None:
        return ConstantNormalizer(0, 1)
    raise ValueError(f"Invalid norm_mode: {norm_mode}")


# --------------------------------------------------------------------------- #
# Channel assemblers
# --------------------------------------------------------------------------- #
def assemble_strip_inputs(bg, ir, r3, r5, context_w=CONTEXT_WIDTH):
    """Strip/reconstruct-mode band + neighbour construction.

    Returns (band, neigh) as tensors:
      band  = [IR, BG]                                          -> (2, H, W)
      neigh = [right_bg, right_ir, left_r5,
               left_bg, left_ir, right_r3]                      -> (6, H, context_w)
    Neighbour context is taken from the ``context_w`` edge columns of the strip.
    """
    bg_t = torch.as_tensor(bg)
    ir_t = torch.as_tensor(ir)
    r3_t = torch.as_tensor(r3)
    r5_t = torch.as_tensor(r5)

    band = torch.stack([ir_t, bg_t], dim=0)

    w = int(context_w)
    left_bg = bg_t[:, :w]
    left_ir = ir_t[:, :w]
    right_r3 = r3_t[:, -w:]
    right_bg = bg_t[:, -w:]
    right_ir = ir_t[:, -w:]
    left_r5 = r5_t[:, :w]

    neigh = torch.stack(
        [right_bg, right_ir, left_r5, left_bg, left_ir, right_r3],
        dim=0,
    )
    return band, neigh


def assemble_patch_inputs_np(bg, ir, r3, r5):
    """Patch/context-mode band + neighbour construction from context patches (numpy).

    ``r3``/``r5`` are (H, W, >=3) context patches; their first three channels are
    used. Returns (x_band, x_neigh) as numpy arrays:
      x_band  = [IR, BG]                                        -> (2, H, W)
      x_neigh = [R5_c0, R5_c1, R5_c2, R3_c0, R3_c1, R3_c2]      -> (6, H, W)
    """
    r3_ctx = np.moveaxis(r3[..., :3], -1, 0)
    r5_ctx = np.moveaxis(r5[..., :3], -1, 0)
    x_neigh = np.concatenate([r5_ctx, r3_ctx], axis=0)
    x_band = np.stack([ir, bg], axis=0)
    return x_band, x_neigh


def assemble_patch_neigh_from_stacked(neighbours):
    """Patch/context-mode neighbour construction from a stacked tensor (training).

    ``neighbours`` is (2, H, Wn, 4) = [RED5, RED3]; the first three channels of each
    are used, giving the same layout as :func:`assemble_patch_inputs_np`:
      [R5_c0, R5_c1, R5_c2, R3_c0, R3_c1, R3_c2]                -> (6, H, Wn)
    """
    neigh = neighbours[..., :3].permute(0, 3, 1, 2)  # (2, 3, Hn, Wn) - drop r4
    return neigh.reshape(-1, neigh.shape[-2], neigh.shape[-1])  # (6, Hn, Wn)
