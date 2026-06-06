# -*- coding: utf-8 -*-
"""
30 SRM (Spatial Rich Model) high-pass residual filters (5×5).

Reference
---------
Fridrich & Kodovský, "Rich Models for Steganalysis of Digital Images",
IEEE Transactions on Information Forensics and Security, 2012.

These are linear prediction error filters of orders 1, 2, 3 in various
spatial directions.  They form the standard preprocessing layer for many
deep steganalysis networks (YeNet, SRNet, YedroudjNet, ZhuNet, etc.).

Usage
-----
    kernels = get_srm_kernels()          # (30, 1, 5, 5)  float32 numpy array
    tensor  = torch.from_numpy(kernels)  # ready for use in nn.Conv2d
"""

import numpy as np

SRM_NUM_FILTERS: int = 30
SRM_KERNEL_SIZE: int = 5

# ZhuNet uses a split bank: 25 3×3 filters + 5 5×5 filters = 30 total
SRM_3x3_NUM: int = 25
SRM_5x5_NUM: int = 5


# ── Raw integer kernels + per-filter normalization divisors ────────────────────
# Each row is a flattened 5×5 filter (25 values).
# Divisors are the sum of absolute values of positive coefficients.

_KERNELS_RAW = np.array([
    # ── Group 1: 1st-order, horizontal / vertical / diagonal (span 3) ──────────
    # s3o1h  symmetric left-right
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  1, -2,  1,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s3o1v  symmetric top-bottom
    [ 0,  0,  0,  0,  0,
      0,  0,  1,  0,  0,
      0,  0, -2,  0,  0,
      0,  0,  1,  0,  0,
      0,  0,  0,  0,  0],
    # s3o1d1  diagonal NW-SE
    [ 0,  0,  0,  0,  0,
      0,  1,  0,  0,  0,
      0,  0, -2,  0,  0,
      0,  0,  0,  1,  0,
      0,  0,  0,  0,  0],
    # s3o1d2  diagonal NE-SW
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  1,  0,
      0,  0, -2,  0,  0,
      0,  1,  0,  0,  0,
      0,  0,  0,  0,  0],

    # ── Group 2: 1st-order, span 5 (4 filters) ──────────────────────────────────
    # s5o1h  symmetric left-right
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
     -1,  0,  0,  0,  1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s5o1v  symmetric top-bottom
    [ 0,  0, -1,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  1,  0,  0],
    # s5o1d1  diagonal NW-SE
    [-1,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  1],
    # s5o1d2  diagonal NE-SW
    [ 0,  0,  0,  0, -1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      1,  0,  0,  0,  0],

    # ── Group 3: 2nd-order (Laplacian-type), span 3 (4 filters) ─────────────────
    # s3o2h  quadratic horizontal
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
     -1,  2, -2,  2, -1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s3o2v  quadratic vertical
    [-1,  0,  0,  0,  0,
      2,  0,  0,  0,  0,
     -2,  0,  0,  0,  0,
      2,  0,  0,  0,  0,
     -1,  0,  0,  0,  0],
    # s3o2d1  2nd-order diagonal NW-SE
    [-1,  0,  0,  0,  0,
      0,  2,  0,  0,  0,
      0,  0, -2,  0,  0,
      0,  0,  0,  2,  0,
      0,  0,  0,  0, -1],
    # s3o2d2  2nd-order diagonal NE-SW
    [ 0,  0,  0,  0, -1,
      0,  0,  0,  2,  0,
      0,  0, -2,  0,  0,
      0,  2,  0,  0,  0,
     -1,  0,  0,  0,  0],

    # ── Group 4: 2nd-order, span 5 (4 filters) ──────────────────────────────────
    # s5o2h
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      1, -4,  6, -4,  1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s5o2v
    [ 1,  0,  0,  0,  0,
     -4,  0,  0,  0,  0,
      6,  0,  0,  0,  0,
     -4,  0,  0,  0,  0,
      1,  0,  0,  0,  0],
    # s5o2d1
    [ 1,  0,  0,  0,  0,
      0, -4,  0,  0,  0,
      0,  0,  6,  0,  0,
      0,  0,  0, -4,  0,
      0,  0,  0,  0,  1],
    # s5o2d2
    [ 0,  0,  0,  0,  1,
      0,  0,  0, -4,  0,
      0,  0,  6,  0,  0,
      0, -4,  0,  0,  0,
      1,  0,  0,  0,  0],

    # ── Group 5: 3rd-order (cubic), span 3 (4 filters) ──────────────────────────
    # s3o3h  cubic horizontal
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
     -1,  3, -3,  1,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s3o3h_r  (mirror)
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  1, -3,  3, -1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # s3o3v
    [ 0,  0, -1,  0,  0,
      0,  0,  3,  0,  0,
      0,  0, -3,  0,  0,
      0,  0,  1,  0,  0,
      0,  0,  0,  0,  0],
    # s3o3v_r  (mirror)
    [ 0,  0,  0,  0,  0,
      0,  0,  1,  0,  0,
      0,  0, -3,  0,  0,
      0,  0,  3,  0,  0,
      0,  0, -1,  0,  0],

    # ── Group 6: 2D edge/cross patterns (4 filters) ──────────────────────────────
    # edge1  Laplacian-like cross
    [ 0,  0,  0,  0,  0,
      0, -1,  2, -1,  0,
      0,  2, -4,  2,  0,
      0, -1,  2, -1,  0,
      0,  0,  0,  0,  0],
    # edge2  full 5×5 KV filter (Kapur–Voelz / XuNet filter)
    [-1,  2, -2,  2, -1,
      2, -6,  8, -6,  2,
     -2,  8,-12,  8, -2,
      2, -6,  8, -6,  2,
     -1,  2, -2,  2, -1],
    # edge3  horizontal 5-tap asymmetric cubic
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
     -1,  2, -6,  2, -1,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # edge4  vertical 5-tap asymmetric cubic
    [ 0,  0, -1,  0,  0,
      0,  0,  2,  0,  0,
      0,  0, -6,  0,  0,
      0,  0,  2,  0,  0,
      0,  0, -1,  0,  0],

    # ── Group 7: cross-channel / mixed (6 filters) ───────────────────────────────
    # square_3  difference from 8-neighbour average (LoG-like)
    [-1, -1, -1, -1, -1,
     -1,  8, -1, -1, -1,
     -1, -1,  9, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1],
    # cross  compass N/S/E/W (span 3 symmetric)
    [ 0,  0, -1,  0,  0,
      0,  0,  2,  0,  0,
     -1,  2, -4,  2, -1,
      0,  0,  2,  0,  0,
      0,  0, -1,  0,  0],
    # h_3  3-tap centred cubic Hx
    [ 0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0, -1,  2, -1,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # v_3  3-tap centred cubic Hy
    [ 0,  0,  0,  0,  0,
      0,  0, -1,  0,  0,
      0,  0,  2,  0,  0,
      0,  0, -1,  0,  0,
      0,  0,  0,  0,  0],
    # diag_3x3 top-left 3×3 gradient
    [-1,  0,  0,  0,  0,
      0,  1,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
    # diag_3x3_r  top-right 3×3 gradient
    [ 0,  0,  0,  0, -1,
      0,  0,  0,  1,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,
      0,  0,  0,  0,  0],
], dtype=np.float32)                          # shape: (30, 25)


def get_srm_kernels() -> np.ndarray:
    """
    Return the 30 SRM high-pass filters as a (30, 1, 5, 5) float32 array.

    Each filter is normalised by the L1 norm of its positive coefficients so
    that the maximum absolute output is ≈1 for a locally constant signal.

    Returns
    -------
    np.ndarray  shape (30, 1, 5, 5), dtype float32
    """
    kernels = _KERNELS_RAW.reshape(SRM_NUM_FILTERS, SRM_KERNEL_SIZE, SRM_KERNEL_SIZE)
    # Normalise by sum of positive elements (standard SRM convention)
    norms = np.maximum(kernels.clip(min=0).sum(axis=(1, 2), keepdims=True), 1.0)
    kernels = kernels / norms
    return kernels[:, np.newaxis, :, :]      # (30, 1, 5, 5)


# ── 25 SRM 3×3 filters (ZhuNet pre-processing bank) ───────────────────────────
# ZhuNet uses a mixed bank: 25 compact 3×3 filters + 5 wider 5×5 filters.
# The 3×3 filters cover 1st/2nd/3rd order differences and 2D patterns that
# fit entirely within a 3×3 neighbourhood.

_SRM_3x3_RAW = np.array([
    # ── 1st order, 4 asymmetric directions ───────────────────────────────────
    [[ 0,  0,  0], [ 0, -1,  1], [ 0,  0,  0]],   # H right
    [[ 0,  0,  0], [-1,  1,  0], [ 0,  0,  0]],   # H left
    [[ 0,  0,  0], [ 0, -1,  0], [ 0,  1,  0]],   # V down
    [[ 0,  1,  0], [ 0, -1,  0], [ 0,  0,  0]],   # V up
    [[ 0,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],   # D1 (NW→SE)
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0,  0]],   # D1 (SE→NW)
    [[ 0,  0,  1], [ 0, -1,  0], [ 0,  0,  0]],   # D2 (NE→SW)
    [[ 0,  0,  0], [ 0, -1,  0], [ 1,  0,  0]],   # D2 (SW→NE)
    # ── 2nd order symmetric, 4 directions ────────────────────────────────────
    [[ 0,  0,  0], [ 1, -2,  1], [ 0,  0,  0]],   # H symmetric
    [[ 0,  1,  0], [ 0, -2,  0], [ 0,  1,  0]],   # V symmetric
    [[ 1,  0,  0], [ 0, -2,  0], [ 0,  0,  1]],   # D1 symmetric
    [[ 0,  0,  1], [ 0, -2,  0], [ 1,  0,  0]],   # D2 symmetric
    # ── 3rd order, 2 directions ───────────────────────────────────────────────
    [[-1,  2, -1], [ 0,  0,  0], [ 0,  0,  0]],   # 2nd-order H (top row)
    [[ 0,  0,  0], [-1,  2, -1], [ 0,  0,  0]],   # 2nd-order H (centre row)
    [[ 0,  0,  0], [ 0,  0,  0], [-1,  2, -1]],   # 2nd-order H (bottom row)
    [[-1,  0,  0], [ 2,  0,  0], [-1,  0,  0]],   # 2nd-order V (left col)
    [[ 0, -1,  0], [ 0,  2,  0], [ 0, -1,  0]],   # 2nd-order V (centre col)
    [[ 0,  0, -1], [ 0,  0,  2], [ 0,  0, -1]],   # 2nd-order V (right col)
    # ── 2D Laplacian / edge ───────────────────────────────────────────────────
    [[ 0,  1,  0], [ 1, -4,  1], [ 0,  1,  0]],   # 4-neighbour Laplacian
    [[ 1,  0,  1], [ 0, -4,  0], [ 1,  0,  1]],   # 4-diagonal Laplacian
    [[ 1,  1,  1], [ 1, -8,  1], [ 1,  1,  1]],   # 8-neighbour Laplacian
    # ── Gradient / Sobel type ─────────────────────────────────────────────────
    [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],   # Sobel V (vertical edges)
    [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],   # Sobel H (horizontal edges)
    [[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]],   # Diagonal emphasis D1
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]],   # Diagonal emphasis D2
], dtype=np.float32)   # (25, 3, 3)


# ── 5 SRM 5×5 filters (ZhuNet pre-processing bank) ────────────────────────────
# These are 5 representative 5×5 high-pass patterns that complement the
# compact 3×3 bank by capturing wider-range residuals.

_SRM_5x5_BANK = np.array([
    # KV filter (used in XuNet, captures wide noise residual)
    [[-1,  2, -2,  2, -1],
     [ 2, -6,  8, -6,  2],
     [-2,  8,-12,  8, -2],
     [ 2, -6,  8, -6,  2],
     [-1,  2, -2,  2, -1]],
    # 5-tap 2nd-order horizontal
    [[ 0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0],
     [ 1, -4,  6, -4,  1],
     [ 0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0]],
    # 5-tap 2nd-order vertical
    [[ 0,  0,  1,  0,  0],
     [ 0,  0, -4,  0,  0],
     [ 0,  0,  6,  0,  0],
     [ 0,  0, -4,  0,  0],
     [ 0,  0,  1,  0,  0]],
    # 5-tap diagonal NW-SE
    [[ 1,  0,  0,  0,  0],
     [ 0, -4,  0,  0,  0],
     [ 0,  0,  6,  0,  0],
     [ 0,  0,  0, -4,  0],
     [ 0,  0,  0,  0,  1]],
    # 5×5 cross (compass)
    [[ 0,  0, -1,  0,  0],
     [ 0,  0,  2,  0,  0],
     [-1,  2, -4,  2, -1],
     [ 0,  0,  2,  0,  0],
     [ 0,  0, -1,  0,  0]],
], dtype=np.float32)   # (5, 5, 5)


def _normalise(kernels: np.ndarray) -> np.ndarray:
    """Divide each kernel by the sum of its positive coefficients (≥1)."""
    norms = np.maximum(kernels.clip(min=0).sum(axis=(-2, -1), keepdims=True), 1.0)
    return kernels / norms


def get_srm_kernels_3x3() -> np.ndarray:
    """Return the 25 SRM 3×3 filters as a (25, 1, 3, 3) float32 array."""
    k = _normalise(_SRM_3x3_RAW)
    return k[:, np.newaxis, :, :]   # (25, 1, 3, 3)


def get_srm_kernels_5x5_bank() -> np.ndarray:
    """Return the 5 ZhuNet 5×5 SRM filters as a (5, 1, 5, 5) float32 array."""
    k = _normalise(_SRM_5x5_BANK)
    return k[:, np.newaxis, :, :]   # (5, 1, 5, 5)
