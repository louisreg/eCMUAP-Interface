"""
Example: Spatial surface animation — multi-filter comparison
=============================================================
Animates the HD-EMG spatial map as a 3-D surface (height + colour =
amplitude) over a peri-stimulus window for six filter conditions:

  0. No spatial filter (interpolated, reference)
  1. NDD  — normal double differential (2-D discrete Laplacian)
  2. LDD  — longitudinal double differential (1-D, along columns)
  3. TDD  — transversal double differential (1-D, along rows)
  4. IB2  — inverse binomial order 2 (isotropic)
  5. IR   — inverse rectangle (isotropic)

Each panel has its own ±|max| colour/height scale so the full dynamic
range of each filter is visible independently.

Output
------
  • Interactive matplotlib animation window
  • ``008_spatial_filter_animation.gif`` saved alongside this script

Pipeline
--------
  raw H32 → LPF + baseline → interpolate to uniform grid → spatial filter
  → animate_snapshot_comparison()

Usage
-----
    python examples/basic/008_spatial_filter_animation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from pathlib import Path

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.probes import make_uniform_probe_from_base
from ecmuap_interface.views.HDemg_view import HDEMGView, animate_snapshot_comparison
import ecmuap_interface.utils.spatial_filters as sp

# ==================================================
# PARAMETERS
# ==================================================
T_PRE    = 1e-3    # s before stimulus
T_POST   = 10e-3   # s after stimulus
SKIP_START = 3     # discard first N trigger events (artifact / warm-up)

FPS      = 12      # animation playback speed (slower for readability)
N_INTERP = 80      # spatial grid resolution (pixels per axis)

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
GIF_FILE  = THIS_FILE.parent.parent / "figures" / "008_spatial_filter_animation.gif"

df_emg = pd.read_hdf(DATA_FILE)
time   = df_emg["time"].values

base_probe = NeuroNexus_H32()
data       = Ripple_to_array(df_emg, base_probe)

trigger = Trigger(data=df_emg["Tr0 "].values, t=time)

emg = EMGData(data=data, time=time, trigger=trigger)
emg.LPF(1_000)
emg.remove_baseline()

# ==================================================
# INTERPOLATE TO UNIFORM GRID
# ==================================================
interp_probe = make_uniform_probe_from_base(base_probe)

hd_base    = HDEMG(emg, base_probe)
emg_interp = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp  = HDEMG(emg_interp, interp_probe)

print(f"Uniform grid: {interp_probe.get_contact_count()} electrodes "
      f"({interp_probe.annotations.get('pitch_um', '?')} µm pitch)")

# ==================================================
# ANIMATION WINDOW: centre on a real trigger event
# ==================================================
events = trigger.get_events()                    # list of event arrays
event_times = events[2]                          # rising-edge times (s)
t_event = event_times[SKIP_START]                # pick first non-skipped event

t_start = t_event - T_PRE
t_end   = t_event + T_POST

print(f"Animation window: [{t_start*1e3:.2f} ms – {t_end*1e3:.2f} ms]  "
      f"(trigger at {t_event*1e3:.2f} ms)")

# ==================================================
# BUILD ONE HDEMGView PER FILTER
# ==================================================
labeled_views = []

for label, kern in FILTERS:
    if kern is None:
        labeled_views.append((label, HDEMGView(hd_interp)))
    else:
        emg_filt = hd_interp.spatial_filter(kern)
        hd_filt  = HDEMG(emg_filt, interp_probe)
        labeled_views.append((label, HDEMGView(hd_filt)))

# ==================================================
# ANIMATE
# ==================================================
print("Precomputing interpolated frames …")

fig, anim = animate_snapshot_comparison(
    views        = labeled_views,
    t_start      = t_start,
    t_end        = t_end,
    fps          = FPS,
    n_interp     = N_INTERP,
    n_cols       = 3,
    cmap         = "RdBu_r",
    show_time    = True,
    shared_scale = False,   # per-filter adaptive scale
    surface      = True,    # 3-D surface: height + colour = amplitude
    elev         = 35,
    azim         = -50,
)

fig.suptitle(
    "HD-EMG spatial map — spatial filter comparison",
    fontsize=13,
    y=1.02,
)

# ==================================================
# SAVE AS GIF
# ==================================================
FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
print(f"Saving animation to {GIF_FILE} …")
anim.save(str(GIF_FILE), writer=PillowWriter(fps=FPS), dpi=72)
print("Done.")

plt.show()
