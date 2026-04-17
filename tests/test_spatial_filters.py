"""
Tests for spatial filter kernels (utils/spatial_filters.py).

Each kernel is validated on simple, hand-verifiable inputs so that
correctness can be confirmed without real data.
"""

import numpy as np
import pytest
from scipy.ndimage import convolve

from ecmuap_interface.utils.spatial_filters import (
    unit_kernel,
    reverse_kernel,
    NDD_kernel,
    LDD_kernel,
    TDD_kernel,
    IB2_kernel,
    IR_kernel,
)

ROWS, COLS = 7, 9

flat      = np.ones((ROWS, COLS))
zeros     = np.zeros((ROWS, COLS))
point     = np.zeros((ROWS, COLS))
point[ROWS // 2, COLS // 2] = 1.0
ramp_col  = np.tile(np.arange(COLS, dtype=float), (ROWS, 1))
ramp_row  = np.tile(np.arange(ROWS, dtype=float)[:, None], (1, COLS))


def _apply(grid_2d: np.ndarray, kern) -> np.ndarray:
    g3 = grid_2d[:, :, np.newaxis].astype(float)
    k3 = kern()[:, :, np.newaxis]
    return convolve(g3, k3, mode="nearest")[:, :, 0]


# ─────────────────────────────────────────────────────────────────────────────
# unit_kernel — identity
# ─────────────────────────────────────────────────────────────────────────────

def test_unit_flat():
    assert np.allclose(_apply(flat, unit_kernel), flat)

def test_unit_point():
    assert np.allclose(_apply(point, unit_kernel), point)


# ─────────────────────────────────────────────────────────────────────────────
# reverse_kernel — sign flip
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_flat():
    assert np.allclose(_apply(flat, reverse_kernel), -flat)

def test_reverse_point():
    assert np.allclose(_apply(point, reverse_kernel), -point)


# ─────────────────────────────────────────────────────────────────────────────
# NDD_kernel — 2-D Laplacian
# ─────────────────────────────────────────────────────────────────────────────

def test_ndd_flat_to_zero():
    assert np.allclose(_apply(flat, NDD_kernel), 0.0)

def test_ndd_zeros_to_zero():
    assert np.allclose(_apply(zeros, NDD_kernel), 0.0)

def test_ndd_point_center():
    out = _apply(point, NDD_kernel)
    r, c = ROWS // 2, COLS // 2
    assert np.isclose(out[r, c], 4.0)

def test_ndd_point_neighbours():
    out = _apply(point, NDD_kernel)
    r, c = ROWS // 2, COLS // 2
    assert np.isclose(out[r - 1, c], -1.0)
    assert np.isclose(out[r + 1, c], -1.0)
    assert np.isclose(out[r, c - 1], -1.0)
    assert np.isclose(out[r, c + 1], -1.0)

def test_ndd_linear_ramp_interior_zero():
    for ramp in (ramp_col, ramp_row):
        out = _apply(ramp, NDD_kernel)
        assert np.allclose(out[1:-1, 1:-1], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LDD_kernel — 1-D double diff along columns
# ─────────────────────────────────────────────────────────────────────────────

def test_ldd_flat_to_zero():
    assert np.allclose(_apply(flat, LDD_kernel), 0.0)

def test_ldd_linear_ramp_interior_zero():
    out = _apply(ramp_row, LDD_kernel)
    assert np.allclose(out[1:-1, :], 0.0)

def test_ldd_parabola_gives_minus2():
    # LDD = -d²/dy²; for f(r)=r², second deriv = 2 → LDD = -2
    out = _apply(ramp_row ** 2, LDD_kernel)
    assert np.allclose(out[1:-1, :], -2.0)


# ─────────────────────────────────────────────────────────────────────────────
# TDD_kernel — 1-D double diff along rows
# ─────────────────────────────────────────────────────────────────────────────

def test_tdd_flat_to_zero():
    assert np.allclose(_apply(flat, TDD_kernel), 0.0)

def test_tdd_linear_ramp_interior_zero():
    out = _apply(ramp_col, TDD_kernel)
    assert np.allclose(out[:, 1:-1], 0.0)

def test_tdd_parabola_gives_minus2():
    out = _apply(ramp_col ** 2, TDD_kernel)
    assert np.allclose(out[:, 1:-1], -2.0)


# ─────────────────────────────────────────────────────────────────────────────
# IB2_kernel — isotropic, weighted 2-D
# ─────────────────────────────────────────────────────────────────────────────

def test_ib2_flat_to_zero():
    assert np.allclose(_apply(flat, IB2_kernel), 0.0, atol=1e-10)

def test_ib2_kernel_sum_zero():
    assert np.isclose(IB2_kernel().sum(), 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# IR_kernel — isotropic rectangle
# ─────────────────────────────────────────────────────────────────────────────

def test_ir_flat_to_zero():
    assert np.allclose(_apply(flat, IR_kernel), 0.0, atol=1e-10)

def test_ir_kernel_sum_zero():
    assert np.isclose(IR_kernel().sum(), 0.0, atol=1e-10)
