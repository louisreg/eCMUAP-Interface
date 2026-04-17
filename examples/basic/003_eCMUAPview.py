"""
Example script: eCMUAP extraction from EMG recording

- Plots individual (non-averaged) eCMUAPs in light color
- Plots averaged eCMUAP per channel on top
- Overlays eCMAP metrics (latency, ttmin, ttmax)
- One distinct color per channel
- Works for mono- or multi-channel EMG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.views.eCMUAP_view import eCMUAPView


# ==================================================
# LOAD DATA
# ==================================================
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
DATA_FILE = BASE_DIR / "../data" / "test_emg.hdf5"

df_emg = pd.read_hdf(DATA_FILE)

# ==================================================
# BUILD TIME VECTOR
# ==================================================
time = df_emg["time"].values  # (n_samples,)

# ==================================================
# BUILD MULTI-CHANNEL EMG ARRAY
# data.shape = (n_channels, n_samples)
# ==================================================
channel_names = ["raw 1", "raw 2", "raw 3"]
data = np.vstack([df_emg[ch].values for ch in channel_names])

print(f"Loaded EMG data: {data.shape[0]} channels, {data.shape[1]} samples")

# ==================================================
# CREATE TRIGGER
# ==================================================
trigger = Trigger(
    data=df_emg["Tr0 "].values,
    t=time,
)

# ==================================================
# CREATE EMGData (CONTINUOUS RECORDING)
# ==================================================
emg = EMGData(
    data=data,
    time=time,
    trigger=trigger,
)

# Simple preprocessing example
emg.LPF(1_000)
emg.remove_baseline()

# ==================================================
# CREATE eCMUAP VIEW
# ==================================================
cmuap = eCMUAPView(emg)

# ==================================================
# EPOCHING PARAMETERS
# ==================================================
t_pre = 1e-3     # 1 ms before stimulation
t_post = 15e-3   # 15 ms after stimulation
skip_start = 3   # skip first epochs
skip_end = 3     # skip last epochs

# ==================================================
# EXTRACT EPOCHS
# ==================================================
epochs = cmuap.epochs(
    t_pre,
    t_post,
    skip_start=skip_start,
    skip_end=skip_end,
)

print("Epochs shape:", epochs.shape)
# (n_events, n_channels, n_samples_epoch)

# ==================================================
# TIME VECTOR FOR EPOCHS
# ==================================================
n_samples_epoch = epochs.shape[-1]
epoch_time = np.linspace(
    -t_pre,
    t_post,
    n_samples_epoch,
    endpoint=False,
)

# ==================================================
# COLORS (ONE PER CHANNEL)
# ==================================================
colors = ["C0", "C1", "C2"]

# ==================================================
# PLOT: INDIVIDUAL eCMUAPs + AVERAGE + METRICS
# ==================================================
fig, ax = plt.subplots(figsize=(8, 4))

n_events, n_channels, _ = epochs.shape

for ch_idx in range(n_channels):

    color = colors[ch_idx % len(colors)]

    # --------------------------------------------------
    # Individual eCMUAPs (light)
    # --------------------------------------------------
    for ev in range(n_events):
        ax.plot(
            epoch_time,
            epochs[ev, ch_idx],
            color=color,
            alpha=0.15,
            linewidth=1,
        )

    # --------------------------------------------------
    # Averaged eCMUAP (bold)
    # --------------------------------------------------
    cmuap.plot_average(
        ch_idx=ch_idx,
        t_pre=t_pre,
        t_post=t_post,
        ax=ax,
        color=color,
        linewidth=2.5,
        label=f"Channel {ch_idx}",
    )

    # --------------------------------------------------
    # Overlay metrics markers
    # --------------------------------------------------
    cmuap.plot_metrics_markers(
        ch_idx=ch_idx,
        t_pre=t_pre,
        t_post=t_post,
        metrics=["latency", "ttmin", "ttmax", "min", "max"],
        ax=ax,
        color=color,
    )

ax.set_title("eCMUAPs: individual trials, average, and metrics")
ax.set_xlabel("Time (s)")
ax.set_ylabel("EMG (µV)")
ax.legend()
plt.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"003_eCMUAPview_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()

# ==================================================
# MONO-CHANNEL EXAMPLE (NO CODE CHANGE)
# ==================================================
emg_mono = EMGData(
    data=data[:1],  # single channel
    time=time,
    trigger=trigger,
)

cmuap_mono = eCMUAPView(emg_mono)
epochs_mono = cmuap_mono.epochs(t_pre, t_post)

print("Mono-channel epochs shape:", epochs_mono.shape)
# (n_events, 1, n_samples_epoch)
