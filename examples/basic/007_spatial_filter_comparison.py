"""
Example: Averaged eCMUAP on probe layout — spatial filter comparison
=====================================================================
Reproduces plot 6 from example 004 (HD-eCMUAP average trace per electrode)
for six conditions:

  0. No spatial filter (interpolated, reference)
  1. NDD  — normal double differential (2-D discrete Laplacian)
  2. LDD  — longitudinal double differential (1-D, along columns)
  3. TDD  — transversal double differential (1-D, along rows)
  4. IB2  — inverse binomial order 2 (isotropic, best SNR per paper)
  5. IR   — inverse rectangle (isotropic)

Pipeline
--------
  raw H32 → LPF + baseline → interpolate to uniform grid → spatial filter
  → plot_avg_eCMUAP()

Reference
---------
  Disselhorst-Klug, Silny & Rau (1997), IEEE Trans. Biomed. Eng., 44(7).
"""

import numpy as np
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

# ==================================================
# PARAMETERS
# ==================================================
T_PRE        = 1e-3    # s before stimulus
T_POST       = 15e-3   # s after stimulus
SKIP_START   = 3
SKIP_END     = 3
T_SNAPSHOT   = 3e-3    # s — for the scale bar reference (not used directly)

FILTERS = [
    ("No filter (reference)", None),
    ("NDD",                   sp.NDD_kernel),
    ("LDD",                   sp.LDD_kernel),
    ("TDD",                   sp.TDD_kernel),
    ("IB2",                   sp.IB2_kernel),
    ("IR",                    sp.IR_kernel),
]

# ==================================================
# LOAD DATA
# ==================================================
THIS_FILE = Path(__file__).resolve()
DATA_FILE = THIS_FILE.parent / "../data/test_emg.hdf5"

df_emg = pd.read_hdf(DATA_FILE)
time   = df_emg["time"].values

base_probe = NeuroNexus_H32()
data       = Ripple_to_array(df_emg, base_probe)

trigger = Trigger(data=df_emg["Tr0 "].values, t=time)

emg = EMGData(data=data, time=time, trigger=trigger)
emg.LPF(1_000)
emg.remove_baseline()

# ==================================================
# INTERPOLATE TO UNIFORM GRID (required for spatial filters)
# ==================================================
interp_probe = make_uniform_probe_from_base(base_probe)

hd_base    = HDEMG(emg, base_probe)
emg_interp = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp  = HDEMG(emg_interp, interp_probe)

print(f"Uniform grid: {interp_probe.get_contact_count()} electrodes "
      f"({interp_probe.annotations.get('pitch_um', '?')} µm pitch)")

# ==================================================
# BUILD ONE HDEMG + HDEMGView PER FILTER
# ==================================================
views = []

for label, kern in FILTERS:
    if kern is None:
        # Reference: no spatial filter
        views.append((label, HDEMGView(hd_interp)))
    else:
        emg_filt = hd_interp.spatial_filter(kern)
        hd_filt  = HDEMG(emg_filt, interp_probe)
        views.append((label, HDEMGView(hd_filt)))

# ==================================================
# PLOT — one figure per filter
# ==================================================
for label, view in views:
    fig, axs = view.plot_avg_eCMUAP(
        t_pre=T_PRE,
        t_post=T_POST,
        skip_start=SKIP_START,
        skip_end=SKIP_END,
        label=False,
        linewidth=1.2,
        color="k",
    )

    view.add_scale_bar(
        fig=fig,
        x_size=5e-3,    # 5 ms
        y_size=2_000,   # 2 mV  (adjust if your signal has a different range)
        x_label="5 ms",
        y_label="2 mV",
        loc="upper right",
    )

    fig.suptitle(
        f"Average eCMUAP — {label}",
        fontsize=13,
        y=1.01,
    )

    plt.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"007_spatial_filter_comparison_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
