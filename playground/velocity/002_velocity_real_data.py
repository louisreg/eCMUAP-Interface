"""
Playground: MUAP propagation velocity on real H32 recordings
=============================================================
Applies the delay-map and cross-correlation CV methods to the
average eCMUAP from real data, for all spatial filter conditions.

Pipeline
--------
  raw H32 → LPF + baseline → interp to uniform grid → spatial filter
  → average eCMUAP → reshape to grid → delay map + xcorr CV

What to look for
----------------
* Delay map:    V-shape (or U-shape) reveals the innervation zone (IZ).
  Rows near IZ have the shortest delay; propagation goes outward.
* xcorr CV map: Positive = propagation in +y, negative = −y.
  Sign flip row ≈ IZ location.
* Filter effect: Does spatial filtering sharpen or distort the
  propagation pattern? Is the CV estimate consistent across filters?

Run
---
    python playground/velocity/002_velocity_real_data.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from pathlib import Path

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.probes import (
    make_uniform_probe_from_base,
    reshape_to_grid,
    get_grid_shape_from_probe,
)
from ecmuap_interface.views.eCMUAP_view import eCMUAPView
import ecmuap_interface.utils.spatial_filters as sp


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

T_PRE      = 2e-3     # s before trigger
T_POST     = 20e-3    # s after trigger (capture full MUAP propagation)
SKIP_START = 3
SKIP_END   = 3

FILTERS = [
    ("No filter",  None),
    ("NDD",        sp.NDD_kernel),
    ("LDD",        sp.LDD_kernel),
    ("TDD",        sp.TDD_kernel),
    ("IB2",        sp.IB2_kernel),
    ("IR",         sp.IR_kernel),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core velocity functions  (identical to 001_propagation_velocity.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_delay_map(data_grid: np.ndarray, fs: float) -> np.ndarray:
    """
    Time-of-negative-peak at each electrode (ms).
    NaN channels (out-of-probe or no signal) → NaN in output.
    """
    rows, cols, n = data_grid.shape
    t_peak = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            sig = data_grid[r, c]
            if not np.any(np.isfinite(sig)):
                continue
            idx = np.nanargmin(sig)
            t_peak[r, c] = idx / fs * 1e3
    return t_peak


def _parabolic_lag(xcorr: np.ndarray, lags: np.ndarray) -> float:
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
    CV between adjacent electrode pairs along `axis` (m/s).
    Positive = MUAP propagates in +axis direction.
    NaN for pairs with missing/noisy channels.
    """
    rows, cols, n = data_grid.shape
    lags_base = correlation_lags(n, n, mode="full").astype(float)

    if axis == 0:
        cv_map  = np.full((rows - 1, cols), np.nan)
        lag_map = np.full((rows - 1, cols), np.nan)
        for c in range(cols):
            for r in range(rows - 1):
                s1, s2 = data_grid[r, c], data_grid[r + 1, c]
                if not (np.any(np.isfinite(s1)) and np.any(np.isfinite(s2))):
                    continue
                s1 = np.nan_to_num(s1)
                s2 = np.nan_to_num(s2)
                xcorr    = correlate(s1, s2, mode="full")
                lag_samp = -_parabolic_lag(xcorr, lags_base)  # neg: s2 delayed → +
                lag_s    = lag_samp / fs
                lag_ms   = lag_s * 1e3
                if abs(lag_ms) > 1e-6:
                    cv_map[r, c]  = (pitch_um * 1e-6) / lag_s
                    lag_map[r, c] = lag_ms
    else:
        cv_map  = np.full((rows, cols - 1), np.nan)
        lag_map = np.full((rows, cols - 1), np.nan)
        for r in range(rows):
            for c in range(cols - 1):
                s1, s2 = data_grid[r, c], data_grid[r, c + 1]
                if not (np.any(np.isfinite(s1)) and np.any(np.isfinite(s2))):
                    continue
                s1 = np.nan_to_num(s1)
                s2 = np.nan_to_num(s2)
                xcorr    = correlate(s1, s2, mode="full")
                lag_samp = -_parabolic_lag(xcorr, lags_base)
                lag_s    = lag_samp / fs
                lag_ms   = lag_s * 1e3
                if abs(lag_ms) > 1e-6:
                    cv_map[r, c]  = (pitch_um * 1e-6) / lag_s
                    lag_map[r, c] = lag_ms
    return cv_map, lag_map


def compute_velocity_map(t_peak: np.ndarray, pitch_um: float):
    """
    Local propagation speed (m/s) + direction from the delay gradient.
    NaN positions are treated as 0 in the gradient (masking them afterwards).
    """
    t_filled = np.where(np.isfinite(t_peak), t_peak, np.nan)
    # np.gradient skips NaN gracefully when using np.ma? No – fill with local mean
    # Simple approach: fill NaN with interpolated neighbours for gradient only
    from scipy.ndimage import generic_filter
    def _nanmean_fill(x):
        v = x[~np.isnan(x)]
        return v.mean() if len(v) > 0 else np.nan
    t_filled2 = np.where(
        np.isfinite(t_filled), t_filled,
        generic_filter(t_filled, _nanmean_fill, size=3, mode="nearest")
    )
    dt_dy, dt_dx = np.gradient(t_filled2, pitch_um, pitch_um)
    denom      = dt_dx**2 + dt_dy**2 + 1e-12
    speed_m_s  = (1.0 / np.sqrt(denom)) * 1e-3   # µm/ms → m/s
    vx_um_ms   = dt_dx / denom
    vy_um_ms   = dt_dy / denom
    # Mask out-of-probe positions
    out = ~np.isfinite(t_peak)
    speed_m_s[out] = np.nan
    vx_um_ms[out]  = np.nan
    vy_um_ms[out]  = np.nan
    return speed_m_s, vx_um_ms, vy_um_ms


# ─────────────────────────────────────────────────────────────────────────────
# Load + preprocess
# ─────────────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
DATA_FILE = THIS_FILE.parent / "../../examples/data/test_emg.hdf5"

import pandas as pd
df_emg = pd.read_hdf(DATA_FILE)
time   = df_emg["time"].values

base_probe = NeuroNexus_H32()
data       = Ripple_to_array(df_emg, base_probe)
trigger    = Trigger(data=df_emg["Tr0 "].values, t=time)

emg = EMGData(data=data, time=time, trigger=trigger)
emg.LPF(1_000)
emg.remove_baseline()

interp_probe = make_uniform_probe_from_base(base_probe)
PITCH_UM     = float(interp_probe.annotations.get("pitch_um", 1726.0))
N_ROWS, N_COLS = get_grid_shape_from_probe(interp_probe)

hd_base    = HDEMG(emg, base_probe)
emg_interp = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp  = HDEMG(emg_interp, interp_probe)

print(f"Uniform grid: {interp_probe.get_contact_count()} electrodes  "
      f"({N_ROWS} rows × {N_COLS} cols, pitch={PITCH_UM:.0f} µm)")
print(f"FS = {emg.fs:.0f} Hz")

cmuap_base = eCMUAPView(emg_interp)


# ─────────────────────────────────────────────────────────────────────────────
# Compute avg eCMUAP + velocity metrics per filter
# ─────────────────────────────────────────────────────────────────────────────

results = []  # (label, avg_grid, t_peak, cv_xcorr, speed, vx, vy)

for label, kern in FILTERS:
    print(f"Processing: {label} …")

    if kern is None:
        emg_filt = emg_interp
    else:
        emg_filt = hd_interp.spatial_filter(kern)

    cmuap_view = eCMUAPView(emg_filt)
    avg = cmuap_view.average(T_PRE, T_POST, SKIP_START, SKIP_END)
    # avg: (n_channels, n_epoch_samples)

    # Reshape to grid
    avg_grid  = reshape_to_grid(avg, interp_probe)           # (rows, cols, n)
    t_peak    = compute_delay_map(avg_grid, emg_filt.fs)     # (rows, cols)
    cv_xcorr, lag_map = compute_xcorr_cv(
        avg_grid, PITCH_UM, emg_filt.fs, axis=0
    )
    speed, vx, vy = compute_velocity_map(t_peak, PITCH_UM)

    finite_cv = cv_xcorr[np.isfinite(cv_xcorr)]
    mean_cv   = np.nanmean(np.abs(finite_cv)) if len(finite_cv) > 0 else np.nan
    print(f"  mean |CV| = {mean_cv:.3f} m/s")

    results.append((label, avg_grid, t_peak, cv_xcorr, speed, vx, vy))


# ─────────────────────────────────────────────────────────────────────────────
# Summary CV table
# ─────────────────────────────────────────────────────────────────────────────

print("\nCV summary (cross-correlation, along fibre axis):")
print(f"  {'Filter':<22}  {'mean |CV|':>10}  {'std':>8}  {'N pairs':>8}")
print("  " + "-" * 54)
for label, _, _, cv_xcorr, _, _, _ in results:
    finite = cv_xcorr[np.isfinite(cv_xcorr)]
    if len(finite) > 0:
        print(f"  {label:<22}  {np.nanmean(np.abs(finite)):>10.3f}  "
              f"{np.nanstd(np.abs(finite)):>8.4f}  {len(finite):>8d}")
    else:
        print(f"  {label:<22}  {'N/A':>10}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Delay maps for all filters
# ─────────────────────────────────────────────────────────────────────────────

n_filt = len(results)
n_cols_plot = 3
n_rows_plot = (n_filt + n_cols_plot - 1) // n_cols_plot

# Find global delay range for consistent colour scale
all_delays = np.concatenate([
    r[2][np.isfinite(r[2])].ravel() for r in results
])
d_vmin, d_vmax = (all_delays.min(), all_delays.max()) if len(all_delays) > 0 else (0, 1)

fig1, axs1 = plt.subplots(n_rows_plot, n_cols_plot,
                           figsize=(5 * n_cols_plot, 4.5 * n_rows_plot))
fig1.suptitle(
    f"Delay map (time of eCMUAP negative peak)\n"
    f"T_PRE={T_PRE*1e3:.0f} ms, T_POST={T_POST*1e3:.0f} ms",
    fontsize=12,
)
axs1_flat = np.array(axs1).ravel()
for ax in axs1_flat[n_filt:]:
    ax.set_visible(False)

for i, (label, avg_grid, t_peak, cv_xcorr, speed, vx, vy) in enumerate(results):
    ax = axs1_flat[i]

    # Probe positions in µm
    pos = interp_probe.contact_positions
    x_u = np.unique(np.round(pos[:, 0], 4))
    y_u = np.unique(np.round(pos[:, 1], 4))

    im = ax.imshow(
        t_peak,
        origin="upper",
        extent=(x_u[0] - PITCH_UM/2, x_u[-1] + PITCH_UM/2,
                y_u[-1] + PITCH_UM/2, y_u[0] - PITCH_UM/2),
        cmap="plasma",
        vmin=d_vmin, vmax=d_vmax,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="delay (ms)", fraction=0.04)

    # Velocity quiver
    x_grid, y_grid = np.meshgrid(x_u, y_u)
    finite_mask = np.isfinite(vx) & np.isfinite(vy)
    if finite_mask.any():
        mag = np.sqrt(vx**2 + vy**2 + 1e-12)
        ax.quiver(
            x_grid[finite_mask], y_grid[finite_mask],
            (vx / mag)[finite_mask], (vy / mag)[finite_mask],
            scale=25, width=0.005, color="white", alpha=0.7,
        )

    # Electrode markers
    ax.scatter(pos[:, 0], pos[:, 1], c="white", s=10,
               edgecolors="gray", linewidths=0.4, zorder=5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

plt.tight_layout()
out1 = "playground/velocity/002_delay_maps.png"
plt.savefig(out1, dpi=120, bbox_inches="tight")
print(f"\nFigure 1 saved → {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Cross-corr CV maps for all filters
# ─────────────────────────────────────────────────────────────────────────────

all_cvs = np.concatenate([
    r[3][np.isfinite(r[3])].ravel() for r in results
])
cv_absmax = (np.percentile(np.abs(all_cvs), 95) * 1.2) if len(all_cvs) > 0 else 10.0

fig2, axs2 = plt.subplots(n_rows_plot, n_cols_plot,
                           figsize=(5 * n_cols_plot, 4.5 * n_rows_plot))
fig2.suptitle("Cross-correlation CV map (m/s)\nPositive = propagating in +y (down), "
              "negative = −y (up)\nSign flip = innervation zone",
              fontsize=12)
axs2_flat = np.array(axs2).ravel()
for ax in axs2_flat[n_filt:]:
    ax.set_visible(False)

for i, (label, avg_grid, t_peak, cv_xcorr, speed, vx, vy) in enumerate(results):
    ax = axs2_flat[i]
    pos = interp_probe.contact_positions
    x_u = np.unique(np.round(pos[:, 0], 4))
    y_u = np.unique(np.round(pos[:, 1], 4))

    # cv_xcorr shape: (rows-1, cols)
    # Plot at midpoints between adjacent row pairs
    y_mid = (y_u[:-1] + y_u[1:]) / 2
    im = ax.imshow(
        cv_xcorr,
        origin="upper",
        extent=(x_u[0] - PITCH_UM/2, x_u[-1] + PITCH_UM/2,
                y_mid[-1] + PITCH_UM/2, y_mid[0] - PITCH_UM/2),
        cmap="RdBu_r",
        vmin=-cv_absmax, vmax=cv_absmax,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="CV (m/s)", fraction=0.04)

    # Sign-flip row = IZ candidate
    for c in range(cv_xcorr.shape[1]):
        col_cv = cv_xcorr[:, c]
        valid  = np.where(np.isfinite(col_cv))[0]
        if len(valid) > 1:
            signs = np.sign(col_cv[valid])
            flip  = valid[np.where(np.diff(signs) != 0)[0]]
            if len(flip) > 0:
                for f in flip:
                    ax.plot(x_u[c], y_mid[f], "r+", markersize=10, markeredgewidth=1.5)

    ax.scatter(pos[:, 0], pos[:, 1], c="white", s=10,
               edgecolors="gray", linewidths=0.4, zorder=5)
    mean_cv = np.nanmean(np.abs(cv_xcorr[np.isfinite(cv_xcorr)]))
    ax.set_title(f"{label}\nmean |CV| = {mean_cv:.2f} m/s", fontsize=9)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

plt.tight_layout()
out2 = "playground/velocity/002_xcorr_cv_maps.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
print(f"Figure 2 saved → {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Bar chart — mean |CV| per filter
# ─────────────────────────────────────────────────────────────────────────────

labels_bar  = [r[0] for r in results]
means_bar   = [np.nanmean(np.abs(r[3][np.isfinite(r[3])])) for r in results]
stds_bar    = [np.nanstd( np.abs(r[3][np.isfinite(r[3])])) for r in results]

fig3, ax3 = plt.subplots(figsize=(8, 4))
bars = ax3.bar(labels_bar, means_bar, yerr=stds_bar, capsize=5,
               color="steelblue", edgecolor="k", linewidth=0.8)
ax3.set_ylabel("Mean |CV| (m/s)")
ax3.set_title("Conduction velocity estimate per spatial filter\n"
              "(cross-correlation, ±1 SD)")
ax3.set_ylim(0, max(means_bar) * 1.5 if means_bar else 10)
ax3.axhline(y=np.nanmean(means_bar), color="r", ls="--", lw=1,
            label=f"grand mean = {np.nanmean(means_bar):.2f} m/s")
ax3.legend(fontsize=9)
plt.tight_layout()
out3 = "playground/velocity/002_cv_summary.png"
plt.savefig(out3, dpi=120, bbox_inches="tight")
print(f"Figure 3 saved → {out3}")

plt.show()
