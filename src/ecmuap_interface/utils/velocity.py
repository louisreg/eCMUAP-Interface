"""
Propagation velocity estimation for HD-EMG.

Functions
---------
compute_delay_map    — time-of-negative-peak per electrode (ms)
compute_xcorr_cv     — inter-electrode conduction velocity via xcorr (m/s)
compute_velocity_map — local speed + direction from the delay gradient

All functions accept NaN channels silently (out-of-probe electrodes after
spatial interpolation are NaN and are simply skipped).

Convention
----------
* Positive CV along axis 0  → MUAP propagates in +row (↓) direction.
* Positive CV along axis 1  → MUAP propagates in +col (→) direction.
* Sign flip row/col marks the innervation zone (IZ).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.ndimage import generic_filter


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_delay_map(avg: np.ndarray, fs: float) -> np.ndarray:
    """
    Time of the negative peak of the average eCMUAP at each electrode.

    Parameters
    ----------
    avg : ndarray, shape (n_channels, n_samples)
        Average eCMUAP, one row per electrode.  NaN channels are skipped.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    delays : ndarray, shape (n_channels,)
        Time of negative peak in **ms**.  NaN for missing channels.
    """
    n_ch, n = avg.shape
    delays = np.full(n_ch, np.nan)
    for ch in range(n_ch):
        sig = avg[ch]
        if np.any(np.isfinite(sig)):
            delays[ch] = int(np.nanargmin(sig)) / fs * 1e3
    return delays


def compute_xcorr_cv(
    avg_grid: np.ndarray,
    pitch_um: float,
    fs: float,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conduction velocity between adjacent electrode pairs along *axis*.

    Uses sub-sample parabolic interpolation of the cross-correlation peak
    for improved accuracy.

    Parameters
    ----------
    avg_grid : ndarray, shape (rows, cols, n_samples)
        Average eCMUAP on a uniform rectangular grid.
        NaN channels are handled gracefully.
    pitch_um : float
        Inter-electrode distance (µm).
    fs : float
        Sampling frequency (Hz).
    axis : {0, 1}
        0 → adjacent row pairs (fibre direction = rows, ↓)
        1 → adjacent column pairs (fibre direction = cols, →)

    Returns
    -------
    cv_map : ndarray
        Conduction velocity in **m/s**.
        Shape (rows-1, cols) for axis=0, (rows, cols-1) for axis=1.
        Positive = propagating in +axis direction.
        NaN for invalid or missing pairs.
    lag_map : ndarray
        Same shape, inter-electrode delay in **ms**.
    """
    rows, cols, n = avg_grid.shape
    lags_base = correlation_lags(n, n, mode="full").astype(float)

    if axis == 0:
        cv_map  = np.full((rows - 1, cols), np.nan)
        lag_map = np.full((rows - 1, cols), np.nan)
        for c in range(cols):
            for r in range(rows - 1):
                s1, s2 = avg_grid[r, c], avg_grid[r + 1, c]
                cv, lag = _xcorr_pair(s1, s2, pitch_um, fs, lags_base)
                cv_map[r, c]  = cv
                lag_map[r, c] = lag
    else:
        cv_map  = np.full((rows, cols - 1), np.nan)
        lag_map = np.full((rows, cols - 1), np.nan)
        for r in range(rows):
            for c in range(cols - 1):
                s1, s2 = avg_grid[r, c], avg_grid[r, c + 1]
                cv, lag = _xcorr_pair(s1, s2, pitch_um, fs, lags_base)
                cv_map[r, c]  = cv
                lag_map[r, c] = lag

    return cv_map, lag_map


def compute_velocity_map(
    delay_grid: np.ndarray,
    pitch_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local propagation speed and direction from the delay gradient.

    Uses the identity  v⃗ = ∇t / |∇t|²  where positions are in µm
    and delays in ms, giving speed in µm/ms = m/s.

    NaN positions (out-of-probe) are filled with nearest-neighbour
    interpolation before gradient computation, then masked out again.

    Parameters
    ----------
    delay_grid : ndarray, shape (rows, cols)
        Per-electrode delay in ms (from :func:`compute_delay_map` reshaped
        to the grid).  NaN for missing electrodes.
    pitch_um : float
        Inter-electrode distance (µm).

    Returns
    -------
    speed : ndarray, shape (rows, cols)
        Local propagation speed in **m/s**.  NaN at missing positions.
    vx : ndarray, shape (rows, cols)
        x-component of the velocity vector (µm/ms).
    vy : ndarray, shape (rows, cols)
        y-component of the velocity vector (µm/ms).
    """
    out_mask = ~np.isfinite(delay_grid)

    # Fill NaN with local mean so np.gradient doesn't propagate them
    def _nn_fill(patch):
        vals = patch[np.isfinite(patch)]
        return vals.mean() if len(vals) else np.nan

    filled = np.where(
        np.isfinite(delay_grid),
        delay_grid,
        generic_filter(delay_grid, _nn_fill, size=3, mode="nearest"),
    )

    # Gradient: dt/dy (axis 0) and dt/dx (axis 1), in ms/µm
    dt_dy, dt_dx = np.gradient(filled, pitch_um, pitch_um)

    denom    = dt_dx**2 + dt_dy**2 + 1e-12      # (ms/µm)²
    vx       = dt_dx / denom                     # µm/ms
    vy       = dt_dy / denom                     # µm/ms
    speed    = np.sqrt(vx**2 + vy**2) * 1e-3    # µm/ms → m/s

    speed[out_mask] = np.nan
    vx[out_mask]    = np.nan
    vy[out_mask]    = np.nan

    return speed, vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parabolic_lag(xcorr: np.ndarray, lags: np.ndarray) -> float:
    """Sub-sample peak of cross-correlation via parabolic interpolation."""
    k = int(np.argmax(xcorr))
    if 0 < k < len(xcorr) - 1:
        denom = xcorr[k - 1] - 2.0 * xcorr[k] + xcorr[k + 1]
        if abs(denom) > 1e-12:
            delta = 0.5 * (xcorr[k - 1] - xcorr[k + 1]) / denom
            return lags[k] + delta * (lags[1] - lags[0])
    return float(lags[k])


def _xcorr_pair(
    s1: np.ndarray,
    s2: np.ndarray,
    pitch_um: float,
    fs: float,
    lags_base: np.ndarray,
) -> tuple[float, float]:
    """
    Compute CV and lag between one adjacent pair.

    scipy.signal.correlate(s1, s2) peaks at lag = −d when s2 is delayed
    by d samples.  We negate so that positive lag = s2 arrives later
    = MUAP propagates from s1 toward s2.
    """
    if not (np.any(np.isfinite(s1)) and np.any(np.isfinite(s2))):
        return np.nan, np.nan

    s1c = np.nan_to_num(s1)
    s2c = np.nan_to_num(s2)

    xcorr    = correlate(s1c, s2c, mode="full")
    lag_samp = -_parabolic_lag(xcorr, lags_base)   # negate: + = s2 delayed
    lag_s    = lag_samp / fs                        # s
    lag_ms   = lag_s * 1e3

    if abs(lag_ms) < 1e-6:
        return np.nan, np.nan

    cv = (pitch_um * 1e-6) / lag_s   # m/s
    return cv, lag_ms
