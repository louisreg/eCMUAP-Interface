"""
Example script: Continuous EMG visualization with trigger overlay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.views.emg_view import EMGView
from ecmuap_interface.views.emg_channel_view import EMGChannelView

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
    data=df_emg["Tr0 "].values,  # raw trigger signal
    t=time,                      # time vector (s)
)

# ==================================================
# CREATE EMGData
# ==================================================
emg = EMGData(
    data=data,
    time=time,
    trigger=trigger,
)

# ==================================================
# CREATE EMG VIEW
# ==================================================
emg_view = EMGView(emg)

# ==================================================
# PLOT RAW EMG WITH TRIGGERS
# ==================================================
fig, ax = plt.subplots(figsize=(10, 4))

emg_view.plot(
    ax=ax,
    raw=True,
    show_triggers=True,
    offset = 0,
    trigger_kwargs=dict(color="k", alpha=0.2, linestyle=":")
)

ax.set_title("Raw EMG (continuous) with stimulation triggers")
plt.tight_layout()

# ==================================================
# APPLY FILTERS AND PLOT FILTERED EMG
# ==================================================

emg.reset()
emg.remove_baseline()
#emg.HPF(10)
emg.LPF(3000)

fig, ax = plt.subplots(figsize=(10, 4))

emg_view.plot(
    ax=ax,
    raw=False,
    show_triggers=True,
    offset = 0,
    trigger_kwargs=dict(color="k", alpha=0.2, linestyle=":")
)

ax.set_title("Filtered EMG (30–3000 Hz) with triggers")
plt.tight_layout()

# --------------------------------------------------
# Single-channel visualization
# --------------------------------------------------
ch_view = EMGChannelView(emg, channel=0)

# Raw EMG
fig, ax = plt.subplots()
ch_view.plot(raw=True, ax=ax)


# Filtered + zoom
emg.reset()
#emg.HPF(10)
emg.LPF(3000)
emg.remove_baseline()

fig, ax = plt.subplots()
ch_view.plot(
    raw=False,
    tlim=(2.5, 3.0),
    ax=ax,
)

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"002_eEMGview_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()