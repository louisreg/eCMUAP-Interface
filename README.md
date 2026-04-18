# eCMUAP-Interface

Python toolkit for processing, analysing, and visualising **evoked Compound Muscle Action Potentials (eCMUAPs)** from surface and high-density EMG (HD-EMG) recordings.

---

## Features

- **Signal filtering** — Butterworth HPF/LPF/BSF, IIR notch, adaptive (NLMS/LMS/RLS) interference removal
- **HD-EMG support** — spatial interpolation to uniform grids and 2-D spatial filtering (NDD, LDD, TDD, IB2, IR kernels)
- **Stimulus-aligned epoching** — extract and average eCMUAPs locked to trigger events, with automatic epoch rejection (amplitude / RMS / correlation)
- **eCMUAP metrics** — latency, duration, peak-to-peak, RMS, time-to-peak (10 % threshold-based)
- **Amplitude maps** — per-electrode RMS, PTP, peak, latency, and duration heatmaps on the probe layout
- **Propagation velocity** — delay map (time-of-negative-peak), cross-correlation CV map, gradient-based velocity field; innervation zone detection
- **Pandas export** — tidy per-electrode `DataFrame` ready for R / SPSS / seaborn downstream analysis
- **Rich visualisations** — multi-channel traces, probe heatmaps, spatial snapshots, metric overlays, animated 3-D surface GIFs
- **Probe library** — NeuroNexus H32 (32-channel) with auto-generated uniform interpolation grids
- **JAX acceleration** *(optional)* — compiled adaptive filters via `jax.lax.scan` + `jax.vmap` (~10–200× speedup)

---

## Installation

Requires Python ≥ 3.12.

```bash
pip install .

# With JAX acceleration for adaptive filters (CPU)
pip install ".[jax]"
```

Or with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

---

## Quick start

### Load data and apply filters

```python
from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.utils.loaders import Ripple_to_array
from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
from ecmuap_interface.utils.trigger import Trigger
import pandas as pd

df   = pd.read_hdf("recording.hdf5")
probe = NeuroNexus_H32()
data  = Ripple_to_array(df, probe)
time  = df["time"].values

trigger = Trigger(df["Tr0 "].values, time)
emg = EMGData(data=data, time=time, trigger=trigger)

emg.LPF(1_000)          # low-pass at 1 kHz
emg.remove_baseline()   # subtract initial baseline pause if present
```

### Extract and average eCMUAPs

```python
from ecmuap_interface.views.eCMUAP_view import eCMUAPView

view = eCMUAPView(emg)

# Reject artefact epochs, then average
avg = view.average(t_pre=1e-3, t_post=20e-3, reject=True, reject_method="amplitude")

# Plot averaged eCMUAP for channel 0
ax = view.plot_average(ch_idx=0, t_pre=1e-3, t_post=20e-3)
view.plot_metrics_markers(ch_idx=0, t_pre=1e-3, t_post=20e-3, ax=ax)
```

### HD-EMG spatial analysis

```python
from ecmuap_interface.core.HD_emg import HDEMG
from ecmuap_interface.utils.probes import make_uniform_probe_from_base
from ecmuap_interface.utils.spatial_filters import IB2_kernel
from ecmuap_interface.views.HDemg_view import HDEMGView

hd           = HDEMG(emg=emg, probe=probe)
uniform_probe = make_uniform_probe_from_base(probe)
emg_interp   = hd.interpolate_to_probe(uniform_probe)

hd_filt = HDEMG(hd_interp.spatial_filter(IB2_kernel), uniform_probe)
view    = HDEMGView(hd_filt)

# Amplitude and timing heatmaps
fig, ax = view.plot_rms_map(t_pre=1e-3, t_post=20e-3)
fig, ax = view.plot_eCMUAP_metric("latency", t_pre=1e-3, t_post=20e-3)

# Propagation velocity
fig, ax = view.plot_delay_map(t_pre=1e-3, t_post=20e-3, show_quiver=True)
fig, ax = view.plot_cv_map(t_pre=1e-3, t_post=20e-3, axis=0, mark_iz=True)
```

### Export metrics to CSV

```python
from ecmuap_interface.utils.export import to_dataframe

df = to_dataframe(view, t_pre=1e-3, t_post=20e-3,
                  filter_label="IB2", reject=True, include_velocity=True)
df.to_csv("results.csv", index=False)
```

---

## Project structure

```text
src/ecmuap_interface/
├── core/
│   ├── emg_data.py          # EMGData   — continuous multi-channel recording
│   ├── eCMUAP.py            # eCMUAP    — single-channel evoked response + metrics
│   └── HD_emg.py            # HDEMG     — EMGData + spatial probe awareness
├── views/
│   ├── emg_view.py          # EMGView          — multi-channel trace plot
│   ├── emg_channel_view.py  # EMGChannelView   — single-channel trace plot
│   ├── eCMUAP_view.py       # eCMUAPView       — epoching, averaging, epoch rejection
│   └── HDemg_view.py        # HDEMGView        — heatmaps, CV maps, animations
├── utils/
│   ├── filters.py           # Butterworth + IIR notch + NLMS/LMS/RLS (NumPy)
│   ├── filters_jax.py       # JAX-accelerated adaptive filters (optional)
│   ├── trigger.py           # Trigger detection and event extraction
│   ├── spatial_filters.py   # 2-D convolution kernels (NDD, LDD, TDD, IB2, IR)
│   ├── velocity.py          # Delay map, cross-correlation CV, gradient velocity field
│   ├── export.py            # Pandas DataFrame export with per-electrode metrics
│   ├── probes.py            # Uniform grid generation, reshape helpers
│   └── loaders.py           # Ripple HDF5 data loader
└── probe_lib/
    └── probes/
        └── NeuroNexus_H32_probe.json
```

---

## Spatial filter kernels

All kernels implement `kernel()` → `np.ndarray`, defined in `utils/spatial_filters.py`.  
Reference: Disselhorst-Klug, Silny & Rau (1997), *IEEE Trans. Biomed. Eng.*, 44(7).

| Name | Type | Description |
| --- | --- | --- |
| `unit_kernel` | identity | Pass-through |
| `reverse_kernel` | sign flip | Invert polarity |
| `NDD_kernel` | 2-D Laplacian | Best general-purpose spatial sharpening |
| `LDD_kernel` | 1-D (column) | Along-column second derivative |
| `TDD_kernel` | 1-D (row) | Along-row second derivative |
| `IB2_kernel` | 2-D weighted | Smooth Laplacian variant, best SNR |
| `IR_kernel` | 2-D isotropic | All-neighbours second derivative |

> Spatial filtering requires a uniform electrode grid — use `make_uniform_probe_from_base()` first.

---

## Examples

End-to-end workflow scripts in `examples/basic/`:

| Script | Topic |
| --- | --- |
| `001_trigger.py` | Trigger detection and visualisation |
| `002_eEMGview.py` | Multi-channel EMG, filtering, baseline removal |
| `003_eCMUAPview.py` | Epoching, averaging, eCMUAP metric overlay |
| `004_HD_emg.py` | HD-EMG probe layout, RMS heatmap, spatial snapshot |
| `005_emg_interpolate.py` | Spatial interpolation to uniform grid |
| `006_spatial_filter.py` | 2-D spatial filtering (NDD) |
| `007_spatial_filter_comparison.py` | All kernels side-by-side on probe layout |
| `008_spatial_filter_animation.py` | Animated 3-D surface GIF across time |
| `009_conduction_velocity.py` | Delay map + cross-correlation CV map |
| `010_epoch_rejection.py` | Automatic artefact rejection before averaging |
| `011_amplitude_maps.py` | RMS / PTP / latency / duration heatmaps |
| `012_export_metrics.py` | Export per-electrode metrics to CSV (pandas) |

Playground / development scripts in `playground/`.

---

## Tests

```bash
pytest tests/          # 54 tests — unit tests for all core modules
```

| File | Tests | Coverage |
| --- | --- | --- |
| `test_emg_data.py` | 10 | `EMGData`, `Trigger` construction and filtering |
| `test_epoch_rejection.py` | 21 | `reject_epochs`, all 3 methods, manual threshold, `average(reject=True)` |
| `test_spatial_filters.py` | 14 | All 7 kernels on flat / point / ramp / parabola inputs |
| `test_velocity.py` | 14 | Delay map, xcorr CV, gradient velocity on synthetic propagating MUAP |

---

## TODO

### Loaders

- [ ] NS5 file loader (Ripple `.ns5` format)
- [ ] Generic loader from NumPy arrays / CSV (no proprietary format required)

### Visualisation

- [ ] Lighter electrode markers in probe plots (smaller, semi-transparent)
- [ ] `highlight_electrode(ch_idx)` — overlay a single electrode across all subplots
- [ ] `plot_epoch_raster()` — show all single-trial epochs as a raster image

### Analysis

- [ ] Multi-session aggregation helper (stack DataFrames from multiple recordings)
- [ ] Multi-session analysis and visualisation

### Code quality

- [ ] Type annotations on public API (`HDEMGView`, `eCMUAPView`)
- [ ] Docstring examples (doctests) for key functions

---

## Data format

The loader expects Ripple (`.hdf5`) recordings with columns:

- `time` — time vector (s)
- `raw 1` … `raw 32` — per-channel raw EMG (µV)
- `Tr0` — binary trigger signal

Channel ordering follows the probe's `device_channel_indices`.

---

## Requirements

| Package | Version |
| --- | --- |
| Python | ≥ 3.12 |
| numpy | ≥ 2.4 |
| scipy | ≥ 1.17 |
| matplotlib | ≥ 3.10 |
| pandas | ≥ 3.0 |
| probeinterface | ≥ 0.3.1 |
| tables | ≥ 3.10 |
| jax / jaxlib | ≥ 0.4 *(optional)* |
