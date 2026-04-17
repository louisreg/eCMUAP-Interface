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

import ecmuap_interface.utils.spatial_filters as sp 

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
# -------------------------------
# APPLY SPATIAL FILTER
# -------------------------------
emg_filt = hd_emg_interp.spatial_filter(
    kernel=sp.NDD_kernel,
    raw=False,
)

hd_emg_filt = HDEMG(emg_filt, interp_probe)
# ==================================================
# CREATE VIEWS
# ==================================================
view_base = HDEMGView(hd_emg)
view_interp = HDEMGView(hd_emg_interp)
view_filt = HDEMGView(hd_emg_filt)
# ==================================================
# TIME INDEX TO COMPARE
# ==================================================
t_snapshot = 3e-3  # 3 ms
t_idx = int(t_snapshot * emg.fs)

# ==================================================
# SHARED COLOR SCALE (important)
# ==================================================
vals_base = emg.data[:, t_idx]
vals_interp = hd_emg_interp.emg.data[:, t_idx]
vals_filt = hd_emg_filt.emg.data[:, t_idx]

vmin = np.nanmin([vals_base.min(), vals_interp.min(), vals_filt.min()])
vmax = np.nanmax([vals_base.max(), vals_interp.max(), vals_filt.max()])

# ==================================================
# FIGURE — GEOMETRY + SNAPSHOTS
# ==================================================
fig, axs = plt.subplots(
    3, 3,
    figsize=(15, 12),
    sharex="col",
    sharey="row",
)

# ==================================================
# ROW 0 — GEOMETRY
# ==================================================
plot_probe(base_probe, ax=axs[0, 0], with_contact_id=False)
axs[0, 0].set_title("Original H32")

plot_probe(interp_probe, ax=axs[0, 1], with_contact_id=False)
axs[0, 1].set_title("Uniform grid")

plot_probe(interp_probe, ax=axs[0, 2], with_contact_id=False)
axs[0, 2].set_title("Uniform grid + NDD")

# ==================================================
# ROW 1 — SNAPSHOT (ELECTRODES / RAW VIEW)
# ==================================================
view_base.plot_snapshot(
    t_idx,
    interpolate=False,
    cmap="viridis",
    ax=axs[1, 0],
)
axs[1, 0].set_title("Base (electrodes)")

view_interp.plot_snapshot(
    t_idx,
    interpolate=False,
    cmap="viridis",
    ax=axs[1, 1],
)
axs[1, 1].set_title("Interpolated data")

view_filt.plot_snapshot(
    t_idx,
    interpolate=False,
    cmap="viridis",
    ax=axs[1, 2],
)
axs[1, 2].set_title("Interpolated + NDD")

# ==================================================
# ROW 2 — SNAPSHOT (INTERPOLATED VIEW)
# ==================================================
view_base.plot_snapshot(
    t_idx,
    interpolate=True,
    cmap="viridis",
    ax=axs[2, 0],
)
axs[2, 0].set_title("Base (interp view)")

view_interp.plot_snapshot(
    t_idx,
    interpolate=True,
    cmap="viridis",
    ax=axs[2, 1],
)
axs[2, 1].set_title("Interpolated (smooth)")

view_filt.plot_snapshot(
    t_idx,
    interpolate=True,
    cmap="viridis",
    ax=axs[2, 2],
)
axs[2, 2].set_title("Interpolated + NDD (smooth)")

# ==================================================
# CLEANUP
# ==================================================
for ax in axs.ravel():
    ax.set_aspect("equal")
    ax.axis("off")

fig.suptitle(
    f"HD-EMG spatial interpolation & filtering (NDD) @ {t_snapshot*1e3:.1f} ms",
    fontsize=15,
)

plt.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"006_spatial_filter_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
