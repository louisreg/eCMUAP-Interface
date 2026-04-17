"""
Example: HD-EMG processing with probeinterface + eCMUAP

- Multi-channel EMG
- Reorder channels using probe.device_channel_indices
- eCMUAP extraction
- Spatial metric heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

from probeinterface import read_probeinterface

from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.eCMUAP import eCMUAP
from ecmuap_interface.views.eCMUAP_view import eCMUAPView


# ==================================================
# PATHS
# ==================================================
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent

DATA_FILE = BASE_DIR / "../../examples/data" / "test_emg.hdf5"
PROBE_FILE = BASE_DIR / "../../examples/data" / "NeuroNexus_H32_probe.json"


# ==================================================
# LOAD PROBE
# ==================================================
probe = read_probeinterface(PROBE_FILE)
probes = read_probeinterface(PROBE_FILE)
probe = probes.probes[0]  # single probe

if probe.device_channel_indices is None:
    raise ValueError("Probe has no device_channel_indices")

print(f"Loaded probe with {probe.get_contact_count()} contacts")


# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_hdf(DATA_FILE)

time = df["time"].values


# ==================================================
# BUILD MULTI-CHANNEL EMG ARRAY (ORDERED BY PROBE)
# ==================================================
device_channels = probe.device_channel_indices  # 0-based

data = []

for ch in device_channels:
    col = f"raw {ch + 1}"  # Ripple uses 1-based indexing
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in DataFrame")
    data.append(df[col].values)

data = np.vstack(data)

print(f"EMG data reordered by probe: {data.shape}")
# shape = (n_contacts, n_samples)


# ==================================================
# CREATE TRIGGER
# ==================================================
trigger = Trigger(
    data=df["Tr0 "].values,
    t=time,
)


# ==================================================
# CREATE EMGData
# ==================================================
emg = EMGData(
    data=data,
    time=time,
    trigger=trigger,
)

# Preprocessing
emg.HPF(30)
emg.LPF(1000)
emg.remove_baseline()


# ==================================================
# eCMUAP VIEW
# ==================================================
cmuap = eCMUAPView(emg)

t_pre = 1e-3     # 1 ms before stim
t_post = 15e-3   # 15 ms after stim

epochs = cmuap.epochs(t_pre, t_post)
avg = cmuap.average(t_pre, t_post)

epoch_time = np.linspace(
    -t_pre,
    t_post,
    avg.shape[1],
    endpoint=False,
)


# ==================================================
# COMPUTE SPATIAL METRIC (HD)
# ==================================================
metric_name = "rms"  # "rms", "latency", "ttpeak", ...

metric_values = np.zeros(emg.n_channels)

for ch in range(emg.n_channels):
    ecmap = eCMUAP(avg[ch], epoch_time)
    metric_values[ch] = getattr(ecmap, metric_name)


# ==================================================
# INTERPOLATE & PLOT HEATMAP
# ==================================================
positions = probe.contact_positions  # (n_channels, 2)

grid_x, grid_y = np.mgrid[
    positions[:, 0].min():positions[:, 0].max():100j,
    positions[:, 1].min():positions[:, 1].max():100j,
]

interp = griddata(
    positions,
    metric_values,
    (grid_x, grid_y),
    method="cubic",
)

fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(
    interp.T,
    origin="lower",
    extent=(
        positions[:, 0].min(),
        positions[:, 0].max(),
        positions[:, 1].min(),
        positions[:, 1].max(),
    ),
    cmap="viridis",
)

# Overlay electrode positions
ax.scatter(
    positions[:, 0],
    positions[:, 1],
    c="white",
    s=40,
    edgecolors="k",
)

ax.set_title(f"HD-eCMUAP {metric_name}")
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
plt.colorbar(im, ax=ax, label=metric_name)

plt.tight_layout()
plt.show()
