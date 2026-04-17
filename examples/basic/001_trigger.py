import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ecmuap_interface.utils.trigger import Trigger

# ==================================================
# RESOLVE ABSOLUTE PATH OF THIS FILE
# ==================================================
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent

data_file = BASE_DIR / "../data" / "test_emg.hdf5"

# ==================================================
# LOAD EMG DATA
# ==================================================
df_emg = pd.read_hdf(data_file)

# Expected columns (example):
# - time        : time vector (s)
# - Tr0         : trigger signal

# ==================================================
# BUILD TRIGGER OBJECT
# ==================================================
trigger_signal = Trigger(
    data=df_emg["Tr0 "].values,   # raw trigger signal
    t=df_emg["time"].values       # time vector (s)
)

# ==================================================
# EXTRACT TRIGGER EVENTS
# ==================================================
event_idx, event_values, event_times = trigger_signal.get_events()

print(f"Detected {len(event_idx)} stimulation events")

# Build stim_events DataFrame (minimal version)
stim_events = pd.DataFrame({
    "sample": event_idx,
    "time_s": event_times,
})

print(stim_events.head())

# ==================================================
# QUICK VISUAL CHECK
# ==================================================
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

# Raw trigger
trigger_signal.plot_raw(ax[0], color="black")
ax[0].set_title("Raw trigger signal")

# Normalized trigger + detected events
trigger_signal.plot_normalized(ax[1], color="gray")
ax[1].plot(
    event_times,
    event_values,
    "ro",
    label="Detected trigger events",
)
ax[1].legend()
ax[1].set_title("Normalized trigger with detected events")

plt.tight_layout()

FIGURES_DIR = THIS_FILE.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
for i, num in enumerate(plt.get_fignums()):
    plt.figure(num).savefig(FIGURES_DIR / f"001_trigger_fig{i+1}.png",
                            dpi=150, bbox_inches="tight")
plt.show()
