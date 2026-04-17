import matplotlib.pyplot as plt
from probeinterface.plotting import plot_probe
import numpy as np

from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.probes import make_uniform_probe_from_base
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.views.HDemg_view import HDEMGView
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.utils.loaders import Ripple_to_array

from pathlib import Path
import pandas as pd

# ==================================================
# LOAD DATA
# ==================================================
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
DATA_FILE = BASE_DIR / "../data" / "test_emg.hdf5"

df_emg = pd.read_hdf(DATA_FILE)
time = df_emg["time"].values

base_probe = NeuroNexus_H32()
data = Ripple_to_array(df_emg, base_probe)

trigger = Trigger(
    data=df_emg["Tr0 "].values,
    t=time,
)

emg = EMGData(
    data=data,
    time=time,
    trigger=trigger,
)

emg.LPF(1000)
emg.remove_baseline()

# ==================================================
# CREATE UNIFORM (INTERPOLATION) PROBE
# ==================================================
interp_probe = make_uniform_probe_from_base(base_probe)

# ==================================================
# CREATE HD-EMG OBJECTS
# ==================================================
hd_emg = HDEMG(emg, base_probe)
emg_interp = hd_emg.interpolate_to_probe(
    target_probe=interp_probe,
    method="cubic",
)
hd_emg_interp = HDEMG(emg_interp, interp_probe)

view_base = HDEMGView(hd_emg)
view_interp = HDEMGView(hd_emg_interp)

# ==================================================
# TIME INDEX TO COMPARE
# ==================================================
t_snapshot = 3e-3  # 3 ms
t_idx = int(t_snapshot * emg.fs)

# shared color scale
values_base = emg.data[:, t_idx]
values_interp = hd_emg_interp.emg.data[:, t_idx]

vmin = np.nanmin([values_base.min(), values_interp.min()])
vmax = np.nanmax([values_base.max(), values_interp.max()])

# ==================================================
# FIGURE
# ==================================================
fig, axs = plt.subplots(
    3, 2,
    figsize=(10, 12),
    sharex="col",
    sharey="row",
)

# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------
plot_probe(base_probe, ax=axs[0, 0], with_contact_id=False)
axs[0, 0].set_title("Original H32 probe")

plot_probe(interp_probe, ax=axs[0, 1], with_contact_id=False)
axs[0, 1].set_title("Uniform interpolation probe")

# --------------------------------------------------
# SNAPSHOT — BASE (NO INTERP)
# --------------------------------------------------
view_base.plot_snapshot(
    t_idx,
    interpolate=False,
    cmap="viridis",
    ax=axs[1, 0],
)
axs[1, 0].set_title("Base HD-EMG (electrodes only)")

view_base.plot_snapshot(
    t_idx,
    interpolate=True,
    cmap="viridis",
    ax=axs[1, 1],
)
axs[1, 1].set_title("Base HD-EMG (interpolated view)")

# --------------------------------------------------
# SNAPSHOT — INTERPOLATED DATA
# --------------------------------------------------
view_interp.plot_snapshot(
    t_idx,
    interpolate=False,
    cmap="viridis",
    ax=axs[2, 0],
)
axs[2, 0].set_title("Interpolated HD-EMG (uniform grid)")

view_interp.plot_snapshot(
    t_idx,
    interpolate=True,
    cmap="viridis",
    ax=axs[2, 1],
)
axs[2, 1].set_title("Interpolated HD-EMG (smoothed)")

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
for ax in axs.ravel():
    ax.set_aspect("equal")

fig.suptitle(
    f"HD-EMG spatial interpolation comparison @ {t_snapshot*1e3:.1f} ms",
    fontsize=14,
)

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
# CREATE VIEWS
# ==================================================
hd_view_base = HDEMGView(hd_emg)
hd_view_interp = HDEMGView(hd_emg_interp)
fig, axs = plt.subplots(
    4,
    4,
    figsize=(18, 12),
    constrained_layout=True,
)

for i, metric in enumerate(metrics):
    row = i // 4
    col = i % 4

    # ---------------- BASE ----------------
    ax = axs[row, col]
    hd_view_base.plot_eCMUAP_metric(
        metric=metric,
        t_pre=t_pre,
        t_post=t_post,
        skip_start=skip_start,
        skip_end=skip_end,
        interpolate=True,
        n_interp=100,
        axis_off=True,
        ax=ax,
    )
    ax.set_title(metric, fontsize=10)

    # ---------------- INTERPOLATED ----------------
    ax = axs[row + 2, col]
    hd_view_interp.plot_eCMUAP_metric(
        metric=metric,
        t_pre=t_pre,
        t_post=t_post,
        skip_start=skip_start,
        skip_end=skip_end,
        interpolate=True,
        n_interp=100,
        axis_off=True,
        ax=ax,
    )

# Row labels
axs[0, 0].set_ylabel("Base probe", fontsize=11)
axs[2, 0].set_ylabel("Interpolated probe", fontsize=11)

fig.suptitle(
    "HD-eCMUAP spatial metrics — base vs interpolated probe",
    fontsize=15,
)

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"005_emg_interpolate_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
