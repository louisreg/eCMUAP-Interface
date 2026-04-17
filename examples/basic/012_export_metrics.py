"""
Example: Export per-electrode eCMUAP metrics to a pandas DataFrame / CSV
=========================================================================
Demonstrates ``ecmuap_interface.utils.export.to_dataframe()``, which
collects per-electrode eCMUAP metrics (amplitude, timing, optional
propagation velocity) into a tidy ``pandas.DataFrame``.

Typical use cases
-----------------
  * Export to CSV for statistical analysis in R / SPSS / seaborn
  * Compare metrics across spatial filters in a single table
  * Correlate electrode position with MUAP amplitude or latency

DataFrame columns
-----------------
  electrode_id   0-based channel index
  x_um, y_um     electrode position (µm)
  rms_uV         RMS amplitude
  ptp_uV         peak-to-peak amplitude
  peak_uV        max absolute amplitude
  latency_ms     onset latency (ms, 10 % threshold)
  duration_ms    eCMUAP duration (ms)
  ttmin_ms       time of negative peak (ms)
  ttmax_ms       time of positive peak (ms)
  delay_ms       [velocity] absolute time of negative peak (ms)
  speed_ms       [velocity] local propagation speed (m/s)
  filter         spatial filter label
  t_pre_s        epoch window parameter
  t_post_s       epoch window parameter

Pipeline
--------
  raw H32 → LPF → baseline → interpolate → [filters] → to_dataframe()
  → concatenate → CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.probes import make_uniform_probe_from_base
from ecmuap_interface.views.HDemg_view import HDEMGView
from ecmuap_interface.utils.export import to_dataframe
import ecmuap_interface.utils.spatial_filters as sp

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
T_PRE      = 1e-3
T_POST     = 20e-3
SKIP_START = 3
SKIP_END   = 3
OUT_CSV    = Path(__file__).parent / "012_metrics.csv"

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
# Interpolate → uniform grid
# ──────────────────────────────────────────────────────────────────────────────
interp_probe = make_uniform_probe_from_base(base_probe)
hd_base      = HDEMG(emg, base_probe)
emg_interp   = hd_base.interpolate_to_probe(interp_probe, method="cubic")
hd_interp    = HDEMG(emg_interp, interp_probe)

# ──────────────────────────────────────────────────────────────────────────────
# Build one HDEMGView per filter
# ──────────────────────────────────────────────────────────────────────────────
FILTERS = [
    ("No filter", None),
    ("IB2",       sp.IB2_kernel),
    ("IR",        sp.IR_kernel),
    ("TDD",       sp.TDD_kernel),
]

views = []
for label, kern in FILTERS:
    if kern is None:
        views.append((label, HDEMGView(hd_interp)))
    else:
        emg_f  = hd_interp.spatial_filter(kern)
        hd_f   = HDEMG(emg_f, interp_probe)
        views.append((label, HDEMGView(hd_f)))

# ──────────────────────────────────────────────────────────────────────────────
# Export metrics — one DataFrame per filter, then concatenate
# ──────────────────────────────────────────────────────────────────────────────
dfs = []
for label, view in views:
    df = to_dataframe(
        view,
        t_pre=T_PRE,
        t_post=T_POST,
        skip_start=SKIP_START,
        skip_end=SKIP_END,
        filter_label=label,
        reject=True,
        reject_method="amplitude",
        include_velocity=True,
    )
    dfs.append(df)
    print(f"\n── {label} ──")
    print(df[["electrode_id", "rms_uV", "ptp_uV", "latency_ms",
              "speed_ms"]].head(8).to_string(index=False))

all_df = pd.concat(dfs, ignore_index=True)
all_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(all_df)} rows → {OUT_CSV}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — RMS distribution per filter (box plot)
# ──────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.boxplot(data=all_df, x="filter", y="rms_uV", hue="filter", ax=ax1,
            palette="Set2", legend=False)
ax1.set_title("RMS amplitude per electrode — spatial filter comparison")
ax1.set_xlabel("Spatial filter")
ax1.set_ylabel("RMS (µV)")
fig1.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Latency vs y position (IB2 only)
# ──────────────────────────────────────────────────────────────────────────────
df_ib2 = all_df[all_df["filter"] == "IB2"].dropna(subset=["latency_ms"])

fig2, ax2 = plt.subplots(figsize=(6, 5))
sc = ax2.scatter(df_ib2["y_um"] / 1e3, df_ib2["latency_ms"],
                 c=df_ib2["rms_uV"], cmap="hot", s=60, edgecolors="k", lw=0.4)
plt.colorbar(sc, ax=ax2, label="RMS (µV)")
ax2.set_xlabel("y position (mm)")
ax2.set_ylabel("Onset latency (ms)")
ax2.set_title("Latency vs electrode y-position (IB2)")
fig2.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Propagation speed heatmap (IB2, on probe layout)
# ──────────────────────────────────────────────────────────────────────────────
df_speed = all_df[all_df["filter"] == "IB2"].dropna(subset=["speed_ms"])

if not df_speed.empty:
    fig3, ax3 = plt.subplots(figsize=(5, 6))
    sc3 = ax3.scatter(
        df_speed["x_um"] / 1e3, df_speed["y_um"] / 1e3,
        c=df_speed["speed_ms"], cmap="plasma",
        s=120, edgecolors="k", lw=0.4,
        vmin=0, vmax=df_speed["speed_ms"].quantile(0.95),
    )
    plt.colorbar(sc3, ax=ax3, label="Speed (m/s)")
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    ax3.set_title("Local propagation speed — IB2 (m/s)")
    ax3.set_aspect("equal")
    fig3.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"012_export_metrics_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
