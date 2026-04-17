"""
Example: Amplitude & timing maps on the HD-EMG probe layout
============================================================
Visualises per-electrode eCMUAP metrics spatially interpolated onto the
probe grid using ``plot_rms_map()``, ``plot_ptp_map()``, and
``plot_eCMUAP_metric()`` (for latency and duration).

Metrics shown
-------------
  * **RMS**      — root-mean-square amplitude (µV)
  * **PTP**      — peak-to-peak amplitude (µV)
  * **Peak**     — maximum absolute amplitude (µV)
  * **Latency**  — eCMUAP onset latency relative to trigger (ms, 10 % threshold)
  * **Duration** — eCMUAP duration (ms)

Two spatial filters are compared side-by-side: IB2 and IR.

Pipeline
--------
  raw H32 → LPF → baseline → interpolate → IB2 / IR → amplitude maps
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.probes import make_uniform_probe_from_base
from ecmuap_interface.views.HDemg_view import HDEMGView
import ecmuap_interface.utils.spatial_filters as sp

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
T_PRE      = 1e-3
T_POST     = 20e-3
SKIP_START = 3
SKIP_END   = 3

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
DATA_FILE = THIS_FILE.parent / "../data/test_emg.hdf5"

df_emg     = pd.read_hdf(DATA_FILE)
time       = df_emg["time"].values
base_probe = NeuroNexus_H32()
data       = Ripple_to_array(df_emg, base_probe)
trigger    = Trigger(data=df_emg["Tr0 "].values, t=time)

emg = EMGData(data=data, time=time, trigger=trigger)
emg.LPF(1_000)
emg.remove_baseline()

# ──────────────────────────────────────────────────────────────────────────────
# Interpolate → uniform grid → two spatial filters
# ──────────────────────────────────────────────────────────────────────────────
interp_probe = make_uniform_probe_from_base(base_probe)
hd_base      = HDEMG(emg, base_probe)
emg_interp   = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp    = HDEMG(emg_interp, interp_probe)

emg_ib2  = hd_interp.spatial_filter(sp.IB2_kernel)
hd_ib2   = HDEMG(emg_ib2, interp_probe)
view_ib2 = HDEMGView(hd_ib2)

emg_ir   = hd_interp.spatial_filter(sp.IR_kernel)
hd_ir    = HDEMG(emg_ir, interp_probe)
view_ir  = HDEMGView(hd_ir)

kw = dict(t_pre=T_PRE, t_post=T_POST,
          skip_start=SKIP_START, skip_end=SKIP_END,
          interpolate=True, n_interp=100)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — RMS map: IB2 vs IR
# ──────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))

view_ib2.plot_rms_map(**kw, cmap="hot", ax=axes1[0])
axes1[0].set_title("RMS amplitude (µV) — IB2")

view_ir.plot_rms_map(**kw, cmap="hot", ax=axes1[1])
axes1[1].set_title("RMS amplitude (µV) — IR")

fig1.suptitle("RMS map", fontsize=13)
fig1.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — PTP map: IB2 vs IR
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

view_ib2.plot_ptp_map(**kw, cmap="hot", ax=axes2[0])
axes2[0].set_title("Peak-to-peak amplitude (µV) — IB2")

view_ir.plot_ptp_map(**kw, cmap="hot", ax=axes2[1])
axes2[1].set_title("Peak-to-peak amplitude (µV) — IR")

fig2.suptitle("Peak-to-peak map", fontsize=13)
fig2.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Five metrics on a single figure (IB2 filter)
# ──────────────────────────────────────────────────────────────────────────────
METRICS = [
    ("rms",      "RMS (µV)",          "hot"),
    ("peak2peak","PTP (µV)",          "hot"),
    ("peak",     "Peak |A| (µV)",     "hot"),
    ("latency",  "Latency (ms)",      "viridis"),
    ("duration", "Duration (ms)",     "cividis"),
]

fig3, axes3 = plt.subplots(1, 5, figsize=(22, 5))
for ax, (metric, label, cmap) in zip(axes3, METRICS):
    view_ib2.plot_eCMUAP_metric(
        **kw,
        metric=metric,
        cmap=cmap,
        ax=ax,
    )
    ax.set_title(label)

fig3.suptitle("eCMUAP metrics — IB2 filter", fontsize=13)
fig3.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"011_amplitude_maps_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
