"""
Example: HD-EMG processing with probeinterface

Pipeline:
- Load Ripple HDF5
- Build HD-EMG array using probe geometry
- Preprocess EMG
- Visualize probe layout
- RMS spatial map
- Instantaneous HD snapshot
- HD-eCMUAP (individual + average)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.views.HDemg_view import HDEMGView


# ==================================================
# LOAD DATA
# ==================================================
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
DATA_FILE = BASE_DIR / "../data" / "test_emg.hdf5"

df_emg = pd.read_hdf(DATA_FILE)
time = df_emg["time"].values  # (n_samples,)

print("Loaded HDF5:", df_emg.shape)


# ==================================================
# LOAD PROBE (probeinterface)
# ==================================================
probe = NeuroNexus_H32()

print("Probe:")
print(" - n_contacts:", probe.get_contact_count())
print(" - device_channel_indices:", probe.device_channel_indices)


# ==================================================
# BUILD HD-EMG ARRAY (ORDERED BY PROBE)
# ==================================================
data = Ripple_to_array(df_emg, probe)
# shape: (n_channels, n_samples)

print("HD-EMG array shape:", data.shape)


# ==================================================
# CREATE TRIGGER
# ==================================================
trigger = Trigger(
    data=df_emg["Tr0 "].values,
    t=time,
)


# ==================================================
# CREATE EMGData (CONTINUOUS)
# ==================================================
emg = EMGData(
    data=data,
    time=time,
    trigger=trigger,
)

# --------------------------------------------------
# SIMPLE PREPROCESSING
# --------------------------------------------------
emg.LPF(1_000)
emg.remove_baseline()


# ==================================================
# CREATE HD-EMG OBJECT + VIEW
# ==================================================
hd_emg = HDEMG(emg, probe)
hd_view = HDEMGView(hd_emg)


# ==================================================
# 1) PROBE GEOMETRY
# ==================================================
fig, ax = plt.subplots(figsize=(4, 6))
hd_view.plot_probe(ax=ax)
ax.set_title("NeuroNexus H32 geometry")
plt.tight_layout()

# ==================================================
# 2) RMS HEATMAP (CONTINUOUS EMG)
# ==================================================
rms = np.sqrt(np.mean(emg.data ** 2, axis=1))

fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

# --- Left: no interpolation (electrode values)
hd_view.plot_metric(
    rms,
    interpolate=False,
    ax=axs[0],
    axis_off=True,
)
axs[0].set_title("HD-EMG RMS (electrodes)")

# --- Right: interpolated heatmap
hd_view.plot_metric(
    rms,
    interpolate=True,
    ax=axs[1],
    axis_off=True,
)
axs[1].set_title("HD-EMG RMS (interpolated)")

plt.tight_layout()

# ==================================================
# 3) INSTANTANEOUS SNAPSHOT
# ==================================================
t_snapshot = 3e-3  # 3 ms
t_idx = int(t_snapshot * emg.fs)

fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

# --- Left: no interpolation (electrode values)
hd_view.plot_snapshot(
    t_idx,
    interpolate=False,
    ax=axs[0],
    axis_off=True,
)
axs[0].set_title(f"Snapshot @ {t_snapshot*1e3:.1f} ms (electrodes)")

# --- Right: interpolated snapshot
hd_view.plot_snapshot(
    t_idx,
    interpolate=True,
    ax=axs[1],
    axis_off=True,
)
axs[1].set_title(f"Snapshot @ {t_snapshot*1e3:.1f} ms (interpolated)")

plt.tight_layout()

# ==================================================
# METRICS TO PLOT
# ==================================================
metrics = [
    "max",
    "min",
    "peak2peak",
    "rms",
    "ttmax",
    "ttmin",
    "latency",
    "duration",
]

t_pre = 1e-3
t_post = 15e-3
skip_start = 3
skip_end = 3
# ==================================================
# FIGURE: 2 x 4 METRIC MAPS
# ==================================================
fig, axs = plt.subplots(
    2,
    4,
    figsize=(16, 7),
    constrained_layout=True,
)

axs = axs.ravel()

for ax, metric in zip(axs, metrics):
    hd_view.plot_eCMUAP_metric(
        metric=metric,
        t_pre=t_pre,
        t_post=t_post,
        skip_start=skip_start,
        skip_end=skip_end,
        interpolate=True,   # False → electrode-only
        n_interp=100,
        axis_off=True,
        ax=ax,
    )
    ax.set_title(metric, fontsize=10)

fig.suptitle(
    "HD-eCMUAP spatial metrics",
    fontsize=14,
)

# ==================================================
# 4) HD-EMG CONTINUOUS SIGNALS (ALL CHANNELS)
# ==================================================
fig, axs = hd_view.plot_data(
    label=True,
    linewidth=0.8,
)

fig.suptitle("HD-EMG continuous signals")
plt.tight_layout()

# ==================================================
# 5) HD-eCMUAP: INDIVIDUAL TRIALS
# ==================================================
fig, axs = hd_view.plot_eCMUAPs(
    t_pre=t_pre,
    t_post=t_post,
    skip_start=skip_start,
    skip_end=skip_end,
    label=False,
    color="k",
    alpha=0.15,
    linewidth=0.8,
)

fig.suptitle("HD-eCMUAP (individual trials)")
plt.tight_layout()

# ==================================================
# 6) HD-eCMUAP: AVERAGE
# ==================================================
fig, axs = hd_view.plot_avg_eCMUAP(
    t_pre=t_pre,
    t_post=t_post,
    skip_start=skip_start,
    skip_end=skip_end,
    label=True,
    linewidth=2.0,
)

hd_view.add_scale_bar(
    fig=fig,
    x_size=5e-3,                  # 5 ms
    y_size=5_000,                   # 100 µV
    x_label="5 ms",
    y_label="5 mV",
)

fig.suptitle("HD-eCMUAP average")
plt.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"004_HD_emg_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()


