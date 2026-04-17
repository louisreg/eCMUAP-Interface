"""
Playground: MUAP propagation velocity estimation
=================================================
Tests three building blocks for estimating muscle fiber conduction
velocity (CV) from a HD-EMG grid, using synthetic data with a
known ground-truth CV so every result can be PASS/FAIL checked.

Methods
-------
1. Delay map   — time-of-peak at each electrode, then linear regression
                 along the fibre axis → CV = pitch / slope
2. Cross-corr  — xcorr between adjacent electrode pairs along the fibre
                 axis, with parabolic sub-sample interpolation of the lag
3. Velocity map — gradient of the delay field → local propagation speed
                  AND direction (quiver overlay on delay heatmap)

Scenarios
---------
A. Unidirectional propagation  (innervation zone at one edge)
B. Bidirectional propagation   (IZ in the middle of the array)

Run
---
    python playground/velocity/001_propagation_velocity.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from scipy.signal import correlate, correlation_lags


# ─────────────────────────────────────────────────────────────────────────────
# Parameters — close to real NeuroNexus H32 uniform grid + Ripple recorder
# ─────────────────────────────────────────────────────────────────────────────

ROWS     = 7          # electrodes along fibre direction (y)
COLS     = 5          # electrodes across fibres (x)
PITCH_UM = 1726.0     # µm — matches H32 uniform-grid pitch
FS       = 30_000     # Hz
T_MS     = 40.0       # total window (ms)
CV_TRUE  = 4.0        # m/s  (= 4000 µm/ms)
SNR      = 15.0       # amplitude signal-to-noise ratio
TOL_PCT  = 7.0        # acceptance threshold for PASS check (%)

# Expected lag between adjacent rows:  1726 µm / 4000 µm/ms = 0.431 ms = 12.9 samples
_EXP_LAG_MS      = PITCH_UM / (CV_TRUE * 1e3)          # ms  (µm / (µm/ms))
_EXP_LAG_SAMPLES = _EXP_LAG_MS * FS / 1e3              # samples (ms * samples/ms)
print(f"[info] Expected inter-electrode lag = {_EXP_LAG_SAMPLES:.1f} samples  "
      f"({_EXP_LAG_MS:.3f} ms)")


# ─────────────────────────────────────────────────────────────────────────────
# MUAP template
# ─────────────────────────────────────────────────────────────────────────────

def make_muap(t_ms: np.ndarray, t0_ms: float = 0.0,
              sigma_ms: float = 0.6, amplitude: float = 1.0) -> np.ndarray:
    """
    Triphasic MUAP: negative 2nd derivative of Gaussian.
    Main (negative) peak at t = t0, flanked by two positive lobes.
    """
    x = (t_ms - t0_ms) / sigma_ms
    return -amplitude * (1.0 - x**2) * np.exp(-0.5 * x**2)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factory
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_data(
    rows: int,
    cols: int,
    pitch_um: float,
    fs: float,
    t_ms: float,
    cv_ms: float,          # m/s
    t_onset_ms: float = 5.0,
    iz_row: int | None = None,
    snr: float = 20.0,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (rows, cols, n_samples) propagating-MUAP array.

    iz_row=None  → IZ at row 0, single-direction propagation (+y)
    iz_row=k     → IZ at row k, bidirectional propagation from that row
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = np.linspace(0, t_ms, int(round(t_ms * 1e-3 * fs)), endpoint=False)
    n = len(t)
    data = np.zeros((rows, cols, n))

    cv_um_ms = cv_ms * 1e3  # µm/ms

    for r in range(rows):
        for c in range(cols):
            dist_um = (abs(r - iz_row) if iz_row is not None else r) * pitch_um
            delay_ms = t_onset_ms + dist_um / cv_um_ms
            data[r, c] = make_muap(t, t0_ms=delay_ms)

    noise_std = 1.0 / snr
    data += rng.normal(0, noise_std, data.shape)
    return t, data


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — delay map (time of negative peak)
# ─────────────────────────────────────────────────────────────────────────────

def compute_delay_map(data_grid: np.ndarray, fs: float) -> np.ndarray:
    """
    Find time of the main (negative) peak at each electrode.

    Returns
    -------
    t_peak : (rows, cols)  in ms
    """
    idx = np.argmin(data_grid, axis=2)
    return idx / fs * 1e3


def fit_cv_from_delays(t_peak: np.ndarray, pitch_um: float,
                        rows_slice=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit delay vs. position (along rows) for each column.

    Returns
    -------
    cv_per_col   : (cols,)  CV in m/s
    slope_per_col: (cols,)  slope in ms/µm (positive = propagating +y)
    """
    rows, cols = t_peak.shape
    if rows_slice is None:
        rows_slice = slice(None)

    y_pos = np.arange(rows)[rows_slice] * pitch_um   # µm
    cv_per_col    = np.full(cols, np.nan)
    slope_per_col = np.full(cols, np.nan)

    for c in range(cols):
        delays = t_peak[rows_slice, c]
        if np.any(np.isnan(delays)):
            continue
        slope, _ = np.polyfit(y_pos, delays, 1)   # ms/µm
        if abs(slope) > 1e-9:
            cv_per_col[c]    = abs(1.0 / (slope * 1e3))   # m/s
            slope_per_col[c] = slope
    return cv_per_col, slope_per_col


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — cross-correlation with parabolic sub-sample interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _parabolic_lag(xcorr: np.ndarray, lags: np.ndarray) -> float:
    """Sub-sample lag via parabolic interpolation around the xcorr peak."""
    k = int(np.argmax(xcorr))
    if 0 < k < len(xcorr) - 1:
        denom = xcorr[k - 1] - 2.0 * xcorr[k] + xcorr[k + 1]
        if abs(denom) > 1e-12:
            delta = 0.5 * (xcorr[k - 1] - xcorr[k + 1]) / denom
            return lags[k] + delta * (lags[1] - lags[0])
    return float(lags[k])


def compute_xcorr_cv(
    data_grid: np.ndarray,
    pitch_um: float,
    fs: float,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate CV between every adjacent electrode pair along `axis`.

    axis=0  → pairs along rows (fibre direction = y)
    axis=1  → pairs along columns

    Returns
    -------
    cv_map  : shape (rows-1, cols) or (rows, cols-1), m/s
              Positive = propagating in +axis direction.
    lag_map : same shape, ms
    """
    rows, cols, n = data_grid.shape
    lags_base = correlation_lags(n, n, mode="full")  # samples

    if axis == 0:
        cv_map  = np.full((rows - 1, cols), np.nan)
        lag_map = np.full((rows - 1, cols), np.nan)
        for c in range(cols):
            for r in range(rows - 1):
                s1, s2 = data_grid[r, c], data_grid[r + 1, c]
                xcorr  = correlate(s1, s2, mode="full")
                # Negate lag: scipy correlate(s1,s2) peaks at k=-d when s2 delayed by d
                # → negate so that positive lag = s2 arrives later = MUAP propagating +axis
                lag_samp = -_parabolic_lag(xcorr, lags_base.astype(float))
                lag_s    = lag_samp / fs
                lag_ms   = lag_s * 1e3
                if abs(lag_ms) > 1e-6:
                    cv_map[r, c]  = (pitch_um * 1e-6) / lag_s   # m/s
                    lag_map[r, c] = lag_ms
    else:
        cv_map  = np.full((rows, cols - 1), np.nan)
        lag_map = np.full((rows, cols - 1), np.nan)
        for r in range(rows):
            for c in range(cols - 1):
                s1, s2 = data_grid[r, c], data_grid[r, c + 1]
                xcorr  = correlate(s1, s2, mode="full")
                lag_samp = -_parabolic_lag(xcorr, lags_base.astype(float))
                lag_s    = lag_samp / fs
                lag_ms   = lag_s * 1e3
                if abs(lag_ms) > 1e-6:
                    cv_map[r, c]  = (pitch_um * 1e-6) / lag_s
                    lag_map[r, c] = lag_ms
    return cv_map, lag_map


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — velocity map from delay gradient
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity_map(t_peak: np.ndarray, pitch_um: float):
    """
    Estimate local propagation velocity vector from the delay gradient.

    v⃗ = ∇t / |∇t|²   where positions are in µm and delays in ms
    → result in µm/ms.  Multiply by 1e-3 to get m/s.

    Returns
    -------
    speed : (rows, cols)  in m/s
    vx    : (rows, cols)  x-component in µm/ms (for quiver)
    vy    : (rows, cols)  y-component in µm/ms
    """
    # np.gradient with spacing → ms/µm
    dt_dy, dt_dx = np.gradient(t_peak, pitch_um, pitch_um)
    denom = dt_dx**2 + dt_dy**2 + 1e-12      # (ms/µm)²
    vx_um_ms = dt_dx / denom                  # µm/ms
    vy_um_ms = dt_dy / denom
    speed_m_s = np.sqrt(vx_um_ms**2 + vy_um_ms**2) * 1e-3   # m/s
    return speed_m_s, vx_um_ms, vy_um_ms


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helper
# ─────────────────────────────────────────────────────────────────────────────

_any_fail = False

def check(label: str, value: float, target: float, tol_pct: float = TOL_PCT):
    global _any_fail
    err_pct = 100.0 * (value - target) / target
    ok = abs(err_pct) <= tol_pct
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  {label}: {value:.3f} m/s  (err = {err_pct:+.1f}%)")
    if not ok:
        _any_fail = True


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO A — unidirectional  (IZ at row 0)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"SCENARIO A — unidirectional propagation  (IZ at row 0)")
print(f"  Grid {ROWS}×{COLS}, pitch={PITCH_UM:.0f} µm, FS={FS} Hz")
print(f"  CV_TRUE = {CV_TRUE} m/s,  SNR = {SNR}")
print(f"{'='*62}")

t_arr, data_A = make_synthetic_data(
    ROWS, COLS, PITCH_UM, FS, T_MS, CV_TRUE, snr=SNR, iz_row=None
)

t_peak_A = compute_delay_map(data_A, FS)

print("\nMethod 1 — delay map (linear fit per column):")
cv_delay_A, slopes_A = fit_cv_from_delays(t_peak_A, PITCH_UM)
for c, cv in enumerate(cv_delay_A):
    check(f"col {c}", cv, CV_TRUE)
check("mean (delay)", float(np.nanmean(cv_delay_A)), CV_TRUE)

print("\nMethod 2 — cross-correlation (row pairs):")
cv_xcorr_A, lag_A = compute_xcorr_cv(data_A, PITCH_UM, FS, axis=0)
check("mean (xcorr)", float(np.nanmean(cv_xcorr_A)), CV_TRUE)
print(f"           std = {np.nanstd(cv_xcorr_A):.4f} m/s")

print("\nMethod 3 — velocity map (delay gradient):")
speed_A, vx_A, vy_A = compute_velocity_map(t_peak_A, PITCH_UM)
check("mean speed", float(np.nanmean(speed_A)), CV_TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B — bidirectional  (IZ in the middle)
# ─────────────────────────────────────────────────────────────────────────────

IZ_ROW = ROWS // 2

print(f"\n{'='*62}")
print(f"SCENARIO B — bidirectional propagation  (IZ at row {IZ_ROW})")
print(f"{'='*62}")

t_arr, data_B = make_synthetic_data(
    ROWS, COLS, PITCH_UM, FS, T_MS, CV_TRUE, snr=SNR, iz_row=IZ_ROW
)

t_peak_B = compute_delay_map(data_B, FS)

print("\nMethod 1 — delay map, fit above/below IZ separately:")
cv_above, _ = fit_cv_from_delays(t_peak_B, PITCH_UM, rows_slice=slice(None, IZ_ROW + 1))
cv_below, _ = fit_cv_from_delays(t_peak_B, PITCH_UM, rows_slice=slice(IZ_ROW, None))
# For the "above" half the delay decreases with row, so slope is negative
# but fit_cv_from_delays already takes abs(1/slope)
check("above IZ (mean)", float(np.nanmean(cv_above)), CV_TRUE)
check("below IZ (mean)", float(np.nanmean(cv_below)), CV_TRUE)

print("\nMethod 2 — cross-correlation:")
cv_xcorr_B, lag_B = compute_xcorr_cv(data_B, PITCH_UM, FS, axis=0)
# Above IZ: MUAP arrives at r+1 before r → lag < 0 → CV < 0 (sign encodes direction)
# Below IZ: MUAP arrives at r+1 after r  → lag > 0 → CV > 0
above_mask = np.arange(ROWS - 1) < IZ_ROW
below_mask = np.arange(ROWS - 1) >= IZ_ROW
check("above IZ |CV| (xcorr)", float(np.nanmean(np.abs(cv_xcorr_B[above_mask, :]))), CV_TRUE)
check("below IZ  CV  (xcorr)", float(np.nanmean(cv_xcorr_B[below_mask, :])),          CV_TRUE)

print("\nMethod 3 — velocity map:")
speed_B, vx_B, vy_B = compute_velocity_map(t_peak_B, PITCH_UM)
# Exclude IZ row: ∂t/∂y ≈ 0 there (V-shape vertex) → speed → ∞
mask_B = np.ones((ROWS, COLS), dtype=bool)
mask_B[IZ_ROW, :] = False
check("mean speed (B, excl. IZ)", float(np.nanmean(speed_B[mask_B])), CV_TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

x_pos = np.arange(COLS) * PITCH_UM
y_pos_rows = np.arange(ROWS) * PITCH_UM
X_grid, Y_grid = np.meshgrid(x_pos, y_pos_rows)

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "MUAP propagation velocity — method comparison on synthetic data\n"
    f"Grid {ROWS}×{COLS}, pitch={PITCH_UM:.0f} µm, FS={FS} Hz, "
    f"CV_TRUE={CV_TRUE} m/s, SNR={SNR}",
    fontsize=11,
)
colors_cols = plt.cm.tab10(np.linspace(0, 1, COLS))

# ── Row 0: Scenario A ────────────────────────────────────────────────────────

# A-left: delay map + velocity field
ax = axs[0, 0]
vmin_d = t_peak_A.min()
vmax_d = t_peak_A.max()
im = ax.imshow(
    t_peak_A,
    origin="upper",
    extent=(x_pos[0] - PITCH_UM/2, x_pos[-1] + PITCH_UM/2,
            y_pos_rows[-1] + PITCH_UM/2, y_pos_rows[0] - PITCH_UM/2),
    cmap="plasma",
    vmin=vmin_d, vmax=vmax_d,
)
plt.colorbar(im, ax=ax, label="delay (ms)")
# Quiver: arrows in direction of propagation, colour = speed
speed_norm_A = (speed_A - speed_A.min()) / (speed_A.max() - speed_A.min() + 1e-9)
ux = vy_A / (np.sqrt(vx_A**2 + vy_A**2) + 1e-9)   # unit y-component → maps to image y
uy = vx_A / (np.sqrt(vx_A**2 + vy_A**2) + 1e-9)   # unit x-component → maps to image x
q = ax.quiver(X_grid, Y_grid, uy, ux, speed_A,
              cmap="cool", scale=30, width=0.005, clim=(0, CV_TRUE * 2))
plt.colorbar(q, ax=ax, label="speed (m/s)")
ax.set_title("A — Delay map + velocity field")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

# A-middle: delay vs position scatter + linear fit
ax = axs[0, 1]
for c in range(COLS):
    ax.scatter(t_peak_A[:, c], y_pos_rows, s=35, color=colors_cols[c],
               label=f"col {c}", zorder=3)
    t_fit = np.array([t_peak_A[:, c].min(), t_peak_A[:, c].max()])
    slope, intercept = np.polyfit(t_peak_A[:, c], y_pos_rows, 1)
    ax.plot(t_fit, slope * t_fit + intercept, "--", color=colors_cols[c], lw=1.2)
ax.axvline(x=t_peak_A.min(), color="gray", lw=0.5, ls=":")
ax.set_xlabel("delay (ms)"); ax.set_ylabel("y position (µm)")
ax.set_title(f"A — Delay vs position\nCV̄ = {np.nanmean(cv_delay_A):.3f} m/s  "
             f"(true = {CV_TRUE})")
ax.legend(fontsize=7, ncol=2)

# A-right: xcorr CV heatmap
ax = axs[0, 2]
cv_vmax = CV_TRUE * 1.5
im2 = ax.imshow(
    cv_xcorr_A,
    origin="upper",
    extent=(x_pos[0] - PITCH_UM/2, x_pos[-1] + PITCH_UM/2,
            y_pos_rows[-2] + PITCH_UM/2, y_pos_rows[0] - PITCH_UM/2),
    cmap="RdBu_r",
    vmin=-cv_vmax, vmax=cv_vmax,
)
plt.colorbar(im2, ax=ax, label="CV (m/s)")
ax.axhline(y=PITCH_UM / 2, color="k", lw=0.5, ls=":")
ax.set_title(f"A — Cross-corr CV map\n"
             f"CV̄ = {np.nanmean(cv_xcorr_A):.3f} m/s  ± {np.nanstd(cv_xcorr_A):.4f}")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

# ── Row 1: Scenario B ────────────────────────────────────────────────────────

# B-left: delay map (V-shape) + velocity field
ax = axs[1, 0]
im3 = ax.imshow(
    t_peak_B,
    origin="upper",
    extent=(x_pos[0] - PITCH_UM/2, x_pos[-1] + PITCH_UM/2,
            y_pos_rows[-1] + PITCH_UM/2, y_pos_rows[0] - PITCH_UM/2),
    cmap="plasma",
)
plt.colorbar(im3, ax=ax, label="delay (ms)")
speed_B2, vx_B2, vy_B2 = compute_velocity_map(t_peak_B, PITCH_UM)
ux_B = vy_B2 / (np.sqrt(vx_B2**2 + vy_B2**2) + 1e-9)
uy_B = vx_B2 / (np.sqrt(vx_B2**2 + vy_B2**2) + 1e-9)
q2 = ax.quiver(X_grid, Y_grid, uy_B, ux_B, speed_B2,
               cmap="cool", scale=30, width=0.005, clim=(0, CV_TRUE * 2))
plt.colorbar(q2, ax=ax, label="speed (m/s)")
ax.axhline(y=IZ_ROW * PITCH_UM, color="r", lw=1.5, ls="--", label=f"IZ (row {IZ_ROW})")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("B — Delay map + velocity field  (IZ = red dashes)")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

# B-middle: V-shape delay vs position
ax = axs[1, 1]
for c in range(COLS):
    ax.scatter(t_peak_B[:, c], y_pos_rows, s=35, color=colors_cols[c],
               label=f"col {c}", zorder=3)
    # draw V-shape fit lines
    y_a = np.arange(0, IZ_ROW + 1) * PITCH_UM
    y_b = np.arange(IZ_ROW, ROWS) * PITCH_UM
    slope_a, int_a = np.polyfit(t_peak_B[:IZ_ROW+1, c], y_a, 1)
    slope_b, int_b = np.polyfit(t_peak_B[IZ_ROW:,   c], y_b, 1)
    t_fit_a = np.array([t_peak_B[:IZ_ROW+1, c].min(), t_peak_B[:IZ_ROW+1, c].max()])
    t_fit_b = np.array([t_peak_B[IZ_ROW:,   c].min(), t_peak_B[IZ_ROW:,   c].max()])
    ax.plot(t_fit_a, slope_a * t_fit_a + int_a, "--", color=colors_cols[c], lw=1.2)
    ax.plot(t_fit_b, slope_b * t_fit_b + int_b, "-",  color=colors_cols[c], lw=1.2)
ax.axhline(y=IZ_ROW * PITCH_UM, color="r", lw=1, ls="--", label="IZ")
ax.set_xlabel("delay (ms)"); ax.set_ylabel("y position (µm)")
ax.set_title(f"B — V-shape delay map  (IZ at row {IZ_ROW})\n"
             f"CV above = {np.nanmean(cv_above):.3f}  "
             f"CV below = {np.nanmean(cv_below):.3f} m/s")
ax.legend(fontsize=7, ncol=2)

# B-right: xcorr CV (sign reversal at IZ)
ax = axs[1, 2]
im4 = ax.imshow(
    cv_xcorr_B,
    origin="upper",
    extent=(x_pos[0] - PITCH_UM/2, x_pos[-1] + PITCH_UM/2,
            y_pos_rows[-2] + PITCH_UM/2, y_pos_rows[0] - PITCH_UM/2),
    cmap="RdBu_r",
    vmin=-cv_vmax, vmax=cv_vmax,
)
plt.colorbar(im4, ax=ax, label="CV (m/s)")
# IZ is between row IZ_ROW-1 and IZ_ROW pairs
iz_y = (IZ_ROW - 0.5) * PITCH_UM
ax.axhline(y=iz_y, color="r", lw=1.5, ls="--", label=f"IZ boundary")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("B — Cross-corr CV map\n"
             "(sign flip reveals IZ location)")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

plt.tight_layout()
out_path = "playground/velocity/001_propagation_velocity.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
if _any_fail:
    print("Some checks FAILED — see above.")
    sys.exit(1)
else:
    print("All checks PASSED.")

plt.show()
