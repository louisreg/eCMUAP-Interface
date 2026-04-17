"""
Example: Conduction velocity estimation on HD-EMG data
=======================================================
Demonstrates the two velocity estimation approaches available in
``ecmuap_interface.utils.velocity``:

  1. **Delay map** — time of the negative MUAP peak per electrode (ms).
     A V-shaped pattern across rows reveals the innervation zone (IZ).
     ``plot_delay_map()`` also shows a quiver field proportional to the
     local propagation direction.

  2. **Cross-correlation CV map** — pair-wise cross-correlation between
     adjacent electrode rows.  Sign convention: positive = MUAP propagates
     in the +row direction (away from IZ); negative = towards IZ.
     ``plot_cv_map()`` marks the sign-flip line (IZ) with red crosses.

Pipeline
--------
  raw H32 → LPF → baseline correction → interpolate to uniform grid
  → IB2 spatial filter → delay map + CV map

Reference
---------
  Farina & Merletti (2004). Methods for estimating muscle fiber conduction
  velocity from surface EMG. J. Neurosci. Methods 134, 199–218.
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
T_PRE      = 1e-3    # s before stimulus
T_POST     = 20e-3   # s after stimulus
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
# Interpolate → uniform grid → IB2 spatial filter
# ──────────────────────────────────────────────────────────────────────────────
interp_probe = make_uniform_probe_from_base(base_probe)
hd_base      = HDEMG(emg, base_probe)
emg_interp   = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp    = HDEMG(emg_interp, interp_probe)

emg_ib2  = hd_interp.spatial_filter(sp.IB2_kernel)
hd_ib2   = HDEMG(emg_ib2, interp_probe)
view_ib2 = HDEMGView(hd_ib2)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Delay map (IB2)
# ──────────────────────────────────────────────────────────────────────────────
fig1, ax1 = view_ib2.plot_delay_map(
    T_PRE, T_POST,
    skip_start=SKIP_START,
    skip_end=SKIP_END,
    cmap="plasma",
    show_quiver=True,
)
ax1.set_title("Delay map (IB2) — time of negative MUAP peak (ms)")
fig1.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Cross-correlation CV map (IB2, along rows)
# ──────────────────────────────────────────────────────────────────────────────
fig2, ax2 = view_ib2.plot_cv_map(
    T_PRE, T_POST,
    skip_start=SKIP_START,
    skip_end=SKIP_END,
    axis=0,
    cmap="RdBu_r",
    mark_iz=True,
)
ax2.set_title("CV map (IB2) — cross-correlation between adjacent rows (m/s)")
fig2.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Side-by-side: delay map vs CV map for two filters
# ──────────────────────────────────────────────────────────────────────────────
emg_ir  = hd_interp.spatial_filter(sp.IR_kernel)
hd_ir   = HDEMG(emg_ir, interp_probe)
view_ir = HDEMGView(hd_ir)

fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

_, ax = view_ib2.plot_delay_map(T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    cmap="plasma", show_quiver=True, ax=axes[0, 0])
axes[0, 0].set_title("IB2 — delay map")

_, ax = view_ib2.plot_cv_map(T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    axis=0, cmap="RdBu_r", mark_iz=True, ax=axes[0, 1])
axes[0, 1].set_title("IB2 — CV map (m/s)")

_, ax = view_ir.plot_delay_map(T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    cmap="plasma", show_quiver=True, ax=axes[1, 0])
axes[1, 0].set_title("IR — delay map")

_, ax = view_ir.plot_cv_map(T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    axis=0, cmap="RdBu_r", mark_iz=True, ax=axes[1, 1])
axes[1, 1].set_title("IR — CV map (m/s)")

fig3.suptitle("Conduction velocity estimation — IB2 vs IR spatial filter",
              fontsize=13)
fig3.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"009_conduction_velocity_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
