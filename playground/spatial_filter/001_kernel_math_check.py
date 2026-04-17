"""
Spatial filter — kernel math sanity check
==========================================
Tests each kernel on simple, hand-verifiable grids so we can confirm
the convolution output matches the expected values WITHOUT needing real data.

All checks print PASS / FAIL.  No matplotlib needed.

Kernels tested
--------------
unit        : identity — output == input
reverse     : negate — output == -input
NDD         : 2-D Laplacian — flat field → 0, point source → spread
LDD / TDD   : 1-D double differential along columns / rows
IB2 / IR    : 2-D double differential variants

Usage
-----
    python playground/spatial_filter/001_kernel_math_check.py
"""

import sys
import numpy as np
from scipy.ndimage import convolve

# Make sure the package is importable when run from the repo root
sys.path.insert(0, "src")

from ecmuap_interface.utils.spatial_filters import (
    unit_kernel,
    reverse_kernel,
    NDD_kernel,
    LDD_kernel,
    TDD_kernel,
    IB2_kernel,
    IR_kernel,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def apply_kernel_to_grid(grid_2d: np.ndarray, kern) -> np.ndarray:
    """Apply a spatial kernel to a 2-D static grid (no time dimension)."""
    # filters_jax / HDEMG use (row, col, time) with kernel[:,:,None].
    # Here we test on pure 2-D grids for readability.
    grid_3d = grid_2d[:, :, np.newaxis].astype(float)
    kernel_3d = kern()[:, :, np.newaxis]
    result_3d = convolve(grid_3d, kernel_3d, mode="nearest")
    return result_3d[:, :, 0]


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}")
    if not condition:
        # Non-zero exit so CI can catch failures
        global _any_fail
        _any_fail = True


_any_fail = False


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

ROWS, COLS = 7, 9

flat   = np.ones((ROWS, COLS))           # uniform field
zeros  = np.zeros((ROWS, COLS))          # all zeros
point  = np.zeros((ROWS, COLS))          # single electrode active
point[ROWS // 2, COLS // 2] = 1.0

ramp_col = np.tile(np.arange(COLS, dtype=float), (ROWS, 1))  # 0,1,2,… per row
ramp_row = np.tile(np.arange(ROWS, dtype=float)[:, None], (1, COLS))  # 0…6 per col


# ---------------------------------------------------------------------------
# unit_kernel  — identity
# ---------------------------------------------------------------------------
print("\n=== unit_kernel ===")
out = apply_kernel_to_grid(flat, unit_kernel)
check("flat field preserved", np.allclose(out, flat))

out = apply_kernel_to_grid(point, unit_kernel)
check("point source preserved", np.allclose(out, point))


# ---------------------------------------------------------------------------
# reverse_kernel  — sign flip
# ---------------------------------------------------------------------------
print("\n=== reverse_kernel ===")
out = apply_kernel_to_grid(flat, reverse_kernel)
check("flat field negated", np.allclose(out, -flat))

out = apply_kernel_to_grid(point, reverse_kernel)
check("point source negated", np.allclose(out, -point))


# ---------------------------------------------------------------------------
# NDD_kernel  — 2-D Laplacian  [[0,-1,0],[-1,4,-1],[0,-1,0]]
#   flat field  → 0 everywhere
#   zeros       → 0 everywhere
#   point at center (away from borders) → center=4, 4 neighbours=-1
# ---------------------------------------------------------------------------
print("\n=== NDD_kernel (2-D Laplacian) ===")
out = apply_kernel_to_grid(flat, NDD_kernel)
check("flat field → all zeros", np.allclose(out, 0.0))

out = apply_kernel_to_grid(zeros, NDD_kernel)
check("zero field → all zeros", np.allclose(out, 0.0))

out = apply_kernel_to_grid(point, NDD_kernel)
r, c = ROWS // 2, COLS // 2
check("point source: center value = 4", np.isclose(out[r, c], 4.0))
check("point source: top    neighbour = -1", np.isclose(out[r - 1, c], -1.0))
check("point source: bottom neighbour = -1", np.isclose(out[r + 1, c], -1.0))
check("point source: left   neighbour = -1", np.isclose(out[r, c - 1], -1.0))
check("point source: right  neighbour = -1", np.isclose(out[r, c + 1], -1.0))
check("point source: diagonal = 0", np.isclose(out[r - 1, c - 1], 0.0))

# Linear ramp: Laplacian of a linear function is 0 (away from borders)
out_ramp_col = apply_kernel_to_grid(ramp_col, NDD_kernel)
check("column ramp: interior Laplacian = 0", np.allclose(out_ramp_col[1:-1, 1:-1], 0.0))

out_ramp_row = apply_kernel_to_grid(ramp_row, NDD_kernel)
check("row ramp: interior Laplacian = 0", np.allclose(out_ramp_row[1:-1, 1:-1], 0.0))


# ---------------------------------------------------------------------------
# LDD_kernel  — 1-D along rows  [[-1],[2],[-1]]  (column differential)
#   flat field → 0, linear column ramp → 0 (second derivative = 0)
# ---------------------------------------------------------------------------
print("\n=== LDD_kernel (1-D double diff, along columns) ===")
out = apply_kernel_to_grid(flat, LDD_kernel)
check("flat field → all zeros", np.allclose(out, 0.0))

out = apply_kernel_to_grid(ramp_row, LDD_kernel)
check("linear column ramp → interior zeros", np.allclose(out[1:-1, :], 0.0))

# Parabola in the column direction: f(r) = r²  →  second deriv = 2
parabola_row = ramp_row ** 2
out = apply_kernel_to_grid(parabola_row, LDD_kernel)
# LDD = -f(r-1) + 2f(r) - f(r+1) = -(r-1)² + 2r² - (r+1)² = -2  (negative 2nd derivative)
check("column parabola: interior LDD = -2", np.allclose(out[1:-1, :], -2.0))


# ---------------------------------------------------------------------------
# TDD_kernel  — 1-D along columns  [[-1, 2, -1]]  (row differential)
# ---------------------------------------------------------------------------
print("\n=== TDD_kernel (1-D double diff, along rows) ===")
out = apply_kernel_to_grid(flat, TDD_kernel)
check("flat field → all zeros", np.allclose(out, 0.0))

out = apply_kernel_to_grid(ramp_col, TDD_kernel)
check("linear row ramp → interior zeros", np.allclose(out[:, 1:-1], 0.0))

parabola_col = ramp_col ** 2
out = apply_kernel_to_grid(parabola_col, TDD_kernel)
# TDD = -f(c-1) + 2f(c) - f(c+1) = -(c-1)² + 2c² - (c+1)² = -2  (negative 2nd derivative)
check("row parabola: interior TDD = -2", np.allclose(out[:, 1:-1], -2.0))


# ---------------------------------------------------------------------------
# IB2_kernel  — weighted 2-D double diff (flat field → 0)
# ---------------------------------------------------------------------------
print("\n=== IB2_kernel ===")
out = apply_kernel_to_grid(flat, IB2_kernel)
check("flat field → all zeros", np.allclose(out, 0.0, atol=1e-10))

# Sum of kernel coefficients should be 0 (preserves DC offset = 0)
coeff_sum = IB2_kernel().sum()
check("kernel coefficients sum to 0", np.isclose(coeff_sum, 0.0, atol=1e-10))


# ---------------------------------------------------------------------------
# IR_kernel  — 2-D isotropic double diff (flat field → 0)
# ---------------------------------------------------------------------------
print("\n=== IR_kernel ===")
out = apply_kernel_to_grid(flat, IR_kernel)
check("flat field → all zeros", np.allclose(out, 0.0, atol=1e-10))

coeff_sum = IR_kernel().sum()
check("kernel coefficients sum to 0", np.isclose(coeff_sum, 0.0, atol=1e-10))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if _any_fail:
    print("Some checks FAILED — see above.")
    sys.exit(1)
else:
    print("All checks PASSED.")
