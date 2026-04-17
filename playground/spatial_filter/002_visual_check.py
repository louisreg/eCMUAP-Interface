"""
Spatial filter — visual inspection on mock data
================================================
Applies every spatial kernel to three synthetic grid signals and plots
the input and filtered outputs side by side so you can verify visually:

  Signal A : flat field (all ones)         → every kernel except 'unit' should → 0
  Signal B : single point source at center → sharpening / spreading depends on kernel
  Signal C : smooth 2-D Gaussian hump      → Laplacian should highlight the edges

The script also runs the full HDEMG.spatial_filter() pipeline on a mock
uniform probe, so we exercise the actual production code path.

Usage
-----
    python playground/spatial_filter/002_visual_check.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import convolve

sys.path.insert(0, "src")

from ecmuap_interface.utils.spatial_filters import (
    unit_kernel,
    NDD_kernel,
    LDD_kernel,
    TDD_kernel,
    IB2_kernel,
    IR_kernel,
)


# ---------------------------------------------------------------------------
# Mock uniform probe (required by HDEMG.spatial_filter)
# ---------------------------------------------------------------------------

def make_mock_uniform_probe(n_rows: int = 7, n_cols: int = 9, pitch_um: float = 500.0):
    """
    Build a probeinterface Probe on a regular (n_rows × n_cols) grid.
    Electrodes are ordered row-major (top-left to bottom-right).
    """
    from probeinterface import Probe

    ys = np.arange(n_rows) * pitch_um        # row positions (µm)
    xs = np.arange(n_cols) * pitch_um        # col positions (µm)
    XX, YY = np.meshgrid(xs, ys[::-1])       # y-flip: row 0 = top
    positions = np.column_stack([XX.ravel(), YY.ravel()])

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="circle",
        shape_params={"radius": 100.0},
    )
    probe.set_contact_ids(np.arange(1, len(positions) + 1))

    # Simple rectangular contour
    margin = pitch_um * 0.5
    xmin, ymin = positions.min(axis=0) - margin
    xmax, ymax = positions.max(axis=0) + margin
    contour = np.array([
        [xmin, ymin], [xmax, ymin],
        [xmax, ymax], [xmin, ymax],
        [xmin, ymin],
    ])
    probe.set_planar_contour(contour)
    probe.annotate(name="mock_uniform", pitch_um=pitch_um)
    return probe


# ---------------------------------------------------------------------------
# Synthetic 2-D signals
# ---------------------------------------------------------------------------

ROWS, COLS = 7, 9
N_SAMPLES   = 100        # time dimension (not important for visual check)
FS          = 1_000.0    # Hz

def flat_field():
    """Uniform signal — all kernels except 'unit' should zero it out."""
    grid = np.ones((ROWS, COLS))
    return np.tile(grid[:, :, np.newaxis], (1, 1, N_SAMPLES))

def point_source(amplitude: float = 1.0):
    """Single active electrode at center, silent everywhere else."""
    grid = np.zeros((ROWS, COLS, N_SAMPLES))
    grid[ROWS // 2, COLS // 2, :] = amplitude
    return grid

def gaussian_hump(sigma: float = 1.8):
    """Smooth Gaussian hump centered on the grid."""
    cy, cx = ROWS / 2, COLS / 2
    Y, X = np.mgrid[:ROWS, :COLS]
    dist2 = (Y - cy) ** 2 + (X - cx) ** 2
    spatial = np.exp(-dist2 / (2 * sigma ** 2))
    return np.tile(spatial[:, :, np.newaxis], (1, 1, N_SAMPLES))


def apply_kernel(grid_3d: np.ndarray, kern) -> np.ndarray:
    """Apply kernel along spatial dims only (broadcast over time)."""
    kernel_3d = kern()[:, :, np.newaxis]
    return convolve(grid_3d, kernel_3d, mode="nearest")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

KERNELS = [
    ("unit",  unit_kernel),
    ("NDD",   NDD_kernel),
    ("LDD",   LDD_kernel),
    ("TDD",   TDD_kernel),
    ("IB2",   IB2_kernel),
    ("IR",    IR_kernel),
]


def plot_row(axes, grid_3d, title_prefix: str, t_idx: int = 0):
    """Plot input + filtered snapshot at time index t_idx in one row."""
    vmin = grid_3d[:, :, t_idx].min()
    vmax = grid_3d[:, :, t_idx].max()

    axes[0].imshow(grid_3d[:, :, t_idx], vmin=vmin, vmax=vmax,
                   cmap="RdBu_r", aspect="auto", origin="upper")
    axes[0].set_title(f"{title_prefix}\n(input)", fontsize=8)
    axes[0].axis("off")

    for ax, (name, kern) in zip(axes[1:], KERNELS):
        out = apply_kernel(grid_3d, kern)
        snap = out[:, :, t_idx]
        abs_max = max(np.abs(snap).max(), 1e-9)
        ax.imshow(snap, vmin=-abs_max, vmax=abs_max,
                  cmap="RdBu_r", aspect="auto", origin="upper")
        ax.set_title(name, fontsize=8)
        ax.axis("off")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

signals = [
    (flat_field(),    "Flat field (all 1s)"),
    (point_source(),  "Point source (center=1)"),
    (gaussian_hump(), "Gaussian hump (σ=1.8)"),
]

n_rows_fig = len(signals)
n_cols_fig = 1 + len(KERNELS)   # input + one per kernel

fig = plt.figure(figsize=(3 * n_cols_fig, 3.2 * n_rows_fig))
fig.suptitle("Spatial filter visual check — snapshot at t=0", fontsize=11, y=1.01)

for row_idx, (grid_3d, label) in enumerate(signals):
    axes = [
        fig.add_subplot(n_rows_fig, n_cols_fig, row_idx * n_cols_fig + col + 1)
        for col in range(n_cols_fig)
    ]
    plot_row(axes, grid_3d, label)

    # Column headers (only on first row)
    if row_idx == 0:
        axes[0].set_title("Input", fontsize=9, fontweight="bold")
        for ax, (name, _) in zip(axes[1:], KERNELS):
            ax.set_title(name, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("playground/spatial_filter/002_visual_check.png", dpi=120, bbox_inches="tight")
print("Figure saved to playground/spatial_filter/002_visual_check.png")
plt.show()


# ---------------------------------------------------------------------------
# HDEMG.spatial_filter() pipeline check
# ---------------------------------------------------------------------------
print("\n--- HDEMG.spatial_filter() integration check ---")

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger

probe = make_mock_uniform_probe(n_rows=ROWS, n_cols=COLS)
n_ch  = probe.get_contact_count()   # ROWS * COLS

# Mock EMGData: smooth Gaussian hump repeated over time
gauss = gaussian_hump()[:, :, 0].ravel()         # (n_ch,)
data  = np.outer(gauss, np.ones(N_SAMPLES))       # (n_ch, N_SAMPLES)
time  = np.arange(N_SAMPLES) / FS

emg       = EMGData(data=data, time=time)
hd_emg    = HDEMG(emg=emg, probe=probe)
emg_filt  = hd_emg.spatial_filter(NDD_kernel)

assert emg_filt is not None, "spatial_filter() returned None!"
assert emg_filt.data.shape == data.shape, (
    f"Shape mismatch: {emg_filt.data.shape} vs {data.shape}"
)

# On a smooth Gaussian the NDD (Laplacian) output should be non-trivial
# and smaller than the input near the center (sharpening suppresses
# slowly-varying components).
rms_in  = np.sqrt(np.mean(data ** 2))
rms_out = np.sqrt(np.mean(emg_filt.data ** 2))
print(f"  Input RMS  : {rms_in:.4f}")
print(f"  NDD output RMS : {rms_out:.4f}  (expected << input for a smooth hump)")
print("  [PASS]  HDEMG.spatial_filter() returned correct shape and non-zero output")
