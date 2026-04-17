"""
Example: Epoch rejection before averaging
==========================================
Demonstrates how to use ``reject_epochs()`` and ``average(reject=True)``
to discard artefact epochs before computing the eCMUAP average.

Three rejection methods are compared:

  * **amplitude** — peak absolute amplitude per epoch (default)
  * **rms**       — RMS amplitude per epoch
  * **correlation** — 1 − correlation with the template (robust to outliers)

The auto-threshold is ``auto_scale × median(scores)`` so no manual tuning
is needed.  A manual threshold can be passed via ``threshold=``.

Pipeline
--------
  raw H32 → LPF → baseline correction → interpolate to uniform grid
  → IB2 filter → reject_epochs() → average()
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
from ecmuap_interface.views.eCMUAP_view import eCMUAPView
import ecmuap_interface.utils.spatial_filters as sp

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
T_PRE      = 1e-3    # s before stimulus
T_POST     = 20e-3   # s after stimulus
SKIP_START = 3
SKIP_END   = 3
CH_PLOT    = 0       # channel to show single-epoch traces for

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
# Interpolate → uniform grid → IB2 filter
# ──────────────────────────────────────────────────────────────────────────────
interp_probe = make_uniform_probe_from_base(base_probe)
hd_base      = HDEMG(emg, base_probe)
emg_interp   = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp    = HDEMG(emg_interp, interp_probe)

emg_ib2  = hd_interp.spatial_filter(sp.IB2_kernel)
hd_ib2   = HDEMG(emg_ib2, interp_probe)

cmuap_view = eCMUAPView(hd_ib2.emg)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Single-channel epoch overlay: all vs rejected
# ──────────────────────────────────────────────────────────────────────────────
all_epochs       = cmuap_view.epochs(T_PRE, T_POST,
                                     skip_start=SKIP_START, skip_end=SKIP_END)
clean_epochs, mask = cmuap_view.reject_epochs(
    T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    method="amplitude",
    return_mask=True,
)
t_epoch = cmuap_view.epoch_time(T_PRE, T_POST, all_epochs.shape[-1]) * 1e3

n_total    = all_epochs.shape[0]
n_rejected = int((~mask).sum())

fig1, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ep in all_epochs[:, CH_PLOT, :]:
    axes[0].plot(t_epoch, ep, color="tab:blue", alpha=0.3, lw=0.6)
axes[0].plot(t_epoch, np.mean(all_epochs[:, CH_PLOT, :], axis=0),
             color="k", lw=2, label="mean (all)")
axes[0].axvline(0, color="r", ls="--", lw=0.8)
axes[0].set_title(f"All epochs (n={n_total})")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude (µV)")
axes[0].legend()

for ep in clean_epochs[:, CH_PLOT, :]:
    axes[1].plot(t_epoch, ep, color="tab:green", alpha=0.3, lw=0.6)
axes[1].plot(t_epoch, np.mean(clean_epochs[:, CH_PLOT, :], axis=0),
             color="k", lw=2, label="mean (clean)")
axes[1].axvline(0, color="r", ls="--", lw=0.8)
axes[1].set_title(f"After rejection (n={n_total - n_rejected} kept, "
                  f"{n_rejected} removed)")
axes[1].set_xlabel("Time (ms)")
axes[1].legend()

fig1.suptitle(f"Epoch rejection — channel {CH_PLOT} — IB2 filter", fontsize=13)
fig1.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Averaged eCMUAP with vs without rejection on the probe layout
# ──────────────────────────────────────────────────────────────────────────────
view_ib2 = HDEMGView(hd_ib2)

fig2, axs2 = view_ib2.plot_avg_eCMUAP(
    T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    reject=False,
    linewidth=1.0, color="tab:blue",
)
fig2.suptitle("Average eCMUAP — no rejection", fontsize=12)
fig2.tight_layout()

fig3, axs3 = view_ib2.plot_avg_eCMUAP(
    T_PRE, T_POST,
    skip_start=SKIP_START, skip_end=SKIP_END,
    reject=True,
    reject_method="amplitude",
    linewidth=1.0, color="tab:green",
)
fig3.suptitle("Average eCMUAP — amplitude rejection", fontsize=12)
fig3.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Compare three rejection methods on one channel
# ──────────────────────────────────────────────────────────────────────────────
methods = ["amplitude", "rms", "correlation"]
colors  = ["tab:orange", "tab:purple", "tab:red"]

fig4, ax4 = plt.subplots(figsize=(8, 4))
avg_all = cmuap_view.average(T_PRE, T_POST,
                             skip_start=SKIP_START, skip_end=SKIP_END,
                             reject=False)
ax4.plot(t_epoch, avg_all[CH_PLOT], color="k", lw=1.5,
         label="no rejection", zorder=5)

for method, col in zip(methods, colors):
    avg_rej = cmuap_view.average(T_PRE, T_POST,
                                 skip_start=SKIP_START, skip_end=SKIP_END,
                                 reject=True, reject_method=method)
    ax4.plot(t_epoch, avg_rej[CH_PLOT], color=col, lw=1.5,
             label=f"{method} rejection")

ax4.axvline(0, color="r", ls="--", lw=0.8)
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Amplitude (µV)")
ax4.set_title(f"Average eCMUAP — channel {CH_PLOT} — rejection method comparison")
ax4.legend()
fig4.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"010_epoch_rejection_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
