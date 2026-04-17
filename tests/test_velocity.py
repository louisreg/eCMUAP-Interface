"""
Tests for utils/velocity.py.

Validates compute_delay_map, compute_xcorr_cv, and compute_velocity_map
on synthetic propagating-MUAP data with a known ground-truth CV.
"""

import numpy as np
import pytest

from ecmuap_interface.utils.velocity import (
    compute_delay_map,
    compute_xcorr_cv,
    compute_velocity_map,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data factory
# ─────────────────────────────────────────────────────────────────────────────

ROWS     = 7
COLS     = 5
PITCH_UM = 1726.0
FS       = 30_000.0
T_MS     = 40.0
CV_TRUE  = 4.0       # m/s
TOL      = 0.10      # 10 % tolerance


def _muap(t_ms, t0_ms, sigma=0.6):
    x = (t_ms - t0_ms) / sigma
    return -(1.0 - x ** 2) * np.exp(-0.5 * x ** 2)


def _make_grid(iz_row=None, snr=30.0):
    t = np.linspace(0, T_MS, int(round(T_MS * 1e-3 * FS)), endpoint=False)
    data = np.zeros((ROWS, COLS, len(t)))
    cv_um_ms = CV_TRUE * 1e3
    for r in range(ROWS):
        for c in range(COLS):
            dist = abs(r - iz_row) * PITCH_UM if iz_row is not None else r * PITCH_UM
            data[r, c] = _muap(t, t0_ms=5.0 + dist / cv_um_ms)
    rng = np.random.default_rng(0)
    data += rng.normal(0, 1.0 / snr, data.shape)
    return t, data


# ─────────────────────────────────────────────────────────────────────────────
# compute_delay_map
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayMap:
    def setup_method(self):
        _, grid = _make_grid()
        # Flatten to (n_ch, n_samples) as compute_delay_map expects
        self.avg = grid.reshape(ROWS * COLS, -1)
        self.grid = grid

    def test_shape(self):
        delays = compute_delay_map(self.avg, FS)
        assert delays.shape == (ROWS * COLS,)

    def test_values_increase_with_row(self):
        """Delay should increase monotonically with row (propagation +y)."""
        delays = compute_delay_map(self.avg, FS)
        d_grid = delays.reshape(ROWS, COLS)
        for c in range(COLS):
            col = d_grid[:, c]
            assert np.all(np.diff(col) > 0), \
                f"Delays not monotone in column {c}: {col}"

    def test_nan_channel_returns_nan(self):
        avg = self.avg.copy()
        avg[0, :] = np.nan
        delays = compute_delay_map(avg, FS)
        assert np.isnan(delays[0])
        assert np.all(np.isfinite(delays[1:]))

    def test_cv_estimate_within_tolerance(self):
        """Linear fit of delay vs row should recover CV within TOL."""
        delays = compute_delay_map(self.avg, FS)
        d_grid = delays.reshape(ROWS, COLS)
        y_pos  = np.arange(ROWS) * PITCH_UM
        for c in range(COLS):
            slope, _ = np.polyfit(y_pos, d_grid[:, c], 1)
            cv_est = abs(1.0 / (slope * 1e3))
            assert abs(cv_est - CV_TRUE) / CV_TRUE < TOL, \
                f"Column {c}: CV={cv_est:.3f} m/s vs {CV_TRUE}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_xcorr_cv
# ─────────────────────────────────────────────────────────────────────────────

class TestXcorrCV:
    def setup_method(self):
        _, self.grid = _make_grid()

    def test_shape_axis0(self):
        cv_map, lag_map = compute_xcorr_cv(self.grid, PITCH_UM, FS, axis=0)
        assert cv_map.shape == (ROWS - 1, COLS)
        assert lag_map.shape == (ROWS - 1, COLS)

    def test_shape_axis1(self):
        cv_map, _ = compute_xcorr_cv(self.grid, PITCH_UM, FS, axis=1)
        assert cv_map.shape == (ROWS, COLS - 1)

    def test_positive_cv_axis0(self):
        """Propagation in +row direction → all CVs should be positive."""
        cv_map, _ = compute_xcorr_cv(self.grid, PITCH_UM, FS, axis=0)
        finite = cv_map[np.isfinite(cv_map)]
        assert np.all(finite > 0)

    def test_mean_cv_within_tolerance(self):
        cv_map, _ = compute_xcorr_cv(self.grid, PITCH_UM, FS, axis=0)
        mean_cv = np.nanmean(cv_map)
        assert abs(mean_cv - CV_TRUE) / CV_TRUE < TOL, \
            f"mean CV={mean_cv:.3f} vs {CV_TRUE}"

    def test_nan_pair_returns_nan(self):
        grid = self.grid.copy()
        grid[2, :, :] = np.nan   # row 2 all NaN
        cv_map, _ = compute_xcorr_cv(grid, PITCH_UM, FS, axis=0)
        # Pairs (1,2) and (2,3) should be NaN
        assert np.all(np.isnan(cv_map[1, :]))
        assert np.all(np.isnan(cv_map[2, :]))

    def test_sign_flip_at_iz(self):
        """Above IZ: CV < 0; below IZ: CV > 0."""
        iz = ROWS // 2
        _, grid_b = _make_grid(iz_row=iz)
        cv_map, _ = compute_xcorr_cv(grid_b, PITCH_UM, FS, axis=0)
        above = cv_map[:iz, :]
        below = cv_map[iz:, :]
        assert np.all(above[np.isfinite(above)] < 0), "above IZ: CV should be < 0"
        assert np.all(below[np.isfinite(below)] > 0), "below IZ: CV should be > 0"


# ─────────────────────────────────────────────────────────────────────────────
# compute_velocity_map
# ─────────────────────────────────────────────────────────────────────────────

class TestVelocityMap:
    def setup_method(self):
        _, grid = _make_grid()
        avg = grid.reshape(ROWS * COLS, -1)
        delays = compute_delay_map(avg, FS)
        self.delay_grid = delays.reshape(ROWS, COLS)

    def test_shape(self):
        speed, vx, vy = compute_velocity_map(self.delay_grid, PITCH_UM)
        assert speed.shape == (ROWS, COLS)
        assert vx.shape    == (ROWS, COLS)
        assert vy.shape    == (ROWS, COLS)

    def test_mean_speed_within_tolerance(self):
        speed, _, _ = compute_velocity_map(self.delay_grid, PITCH_UM)
        mean_speed = np.nanmean(speed)
        assert abs(mean_speed - CV_TRUE) / CV_TRUE < TOL, \
            f"mean speed={mean_speed:.3f} vs {CV_TRUE}"

    def test_nan_input_returns_nan(self):
        dg = self.delay_grid.copy()
        dg[0, 0] = np.nan
        speed, vx, vy = compute_velocity_map(dg, PITCH_UM)
        assert np.isnan(speed[0, 0])

    def test_propagation_direction(self):
        """vy (y-component) should dominate over vx for +y propagation."""
        _, vx, vy = compute_velocity_map(self.delay_grid, PITCH_UM)
        finite = np.isfinite(vx) & np.isfinite(vy)
        mean_vy = np.mean(vy[finite])
        mean_vx = np.mean(vx[finite])
        assert abs(mean_vy) > abs(mean_vx), \
            "Expected vy > vx for propagation along rows"
