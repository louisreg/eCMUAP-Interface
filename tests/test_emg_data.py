"""
Tests for core EMGData and Trigger classes.
"""

import numpy as np
import pytest

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.utils.trigger import Trigger


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

FS       = 10_000.0
DURATION = 1.0      # s
N_CH     = 8


def _make_emg(seed=0):
    rng     = np.random.default_rng(seed)
    n       = int(DURATION * FS)
    data    = rng.normal(0, 100.0, (N_CH, n))
    time    = np.arange(n) / FS
    return EMGData(data=data, time=time)


def _make_trigger(n_events=10):
    n    = int(DURATION * FS)
    time = np.arange(n) / FS
    trig = np.zeros(n)
    # Place events every 80 ms, starting at 50 ms
    for k in range(n_events):
        idx = int((0.05 + k * 0.08) * FS)
        if idx < n:
            trig[idx : idx + 3] = 5.0
    return Trigger(data=trig, t=time)


# ─────────────────────────────────────────────────────────────────────────────
# EMGData construction
# ─────────────────────────────────────────────────────────────────────────────

def test_emg_shape():
    emg = _make_emg()
    assert emg.n_channels == N_CH
    assert emg.n_samples  == int(DURATION * FS)


def test_emg_fs():
    emg = _make_emg()
    assert np.isclose(emg.fs, FS, rtol=1e-3)


def test_emg_data_2d_required():
    """EMGData must reject non-2D data arrays."""
    with pytest.raises(ValueError):
        EMGData(data=np.zeros(100), time=np.arange(100) / FS)


# ─────────────────────────────────────────────────────────────────────────────
# LPF
# ─────────────────────────────────────────────────────────────────────────────

def test_lpf_attenuates_high_freq():
    """After LPF(100 Hz), a 500 Hz sinusoid should be strongly attenuated."""
    n    = int(DURATION * FS)
    time = np.arange(n) / FS
    freq = 500.0
    data = np.tile(np.sin(2 * np.pi * freq * time), (N_CH, 1)) * 100.0
    emg  = EMGData(data=data.copy(), time=time)
    emg.LPF(100.0)
    rms_before = np.sqrt(np.mean(data ** 2))
    rms_after  = np.sqrt(np.mean(emg.data ** 2))
    assert rms_after < 0.1 * rms_before, \
        f"LPF did not attenuate: before={rms_before:.1f}, after={rms_after:.1f}"


def test_lpf_preserves_low_freq():
    """A 10 Hz sinusoid should pass through a 100 Hz LPF mostly unchanged."""
    n    = int(DURATION * FS)
    time = np.arange(n) / FS
    freq = 10.0
    data = np.tile(np.sin(2 * np.pi * freq * time), (N_CH, 1)) * 100.0
    emg  = EMGData(data=data.copy(), time=time)
    emg.LPF(100.0)
    rms_before = np.sqrt(np.mean(data ** 2))
    rms_after  = np.sqrt(np.mean(emg.data ** 2))
    assert rms_after > 0.8 * rms_before, \
        f"LPF attenuated too much: before={rms_before:.1f}, after={rms_after:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# remove_baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_baseline_reduces_mean():
    """remove_baseline subtracts the mean computed between trigger[0] and trigger[1]."""
    rng  = np.random.default_rng(42)
    # 5 s signal so we have a large first inter-event gap (> 0.5 s)
    duration = 5.0
    n    = int(duration * FS)
    time = np.arange(n) / FS
    # Add a large DC offset
    data = rng.normal(0, 10.0, (N_CH, n)) + 500.0

    # Build a trigger with an abnormally large first gap (2 s) then regular 100 ms events
    trig = np.zeros(n)
    # First event at t=2 s, then events every 100 ms from t=2.5 s onward
    first_idx = int(2.0 * FS)
    trig[first_idx] = 5.0
    for k in range(20):
        idx = int((2.5 + k * 0.1) * FS)
        if idx < n:
            trig[idx] = 5.0

    trigger = Trigger(data=trig, t=time)
    emg = EMGData(data=data, time=time, trigger=trigger)

    mean_before = np.abs(np.mean(emg.data))
    applied = emg.remove_baseline()
    mean_after = np.abs(np.mean(emg.data))

    assert applied, "remove_baseline should have been applied"
    assert mean_after < mean_before * 0.01, \
        "remove_baseline should eliminate DC offset"


# ─────────────────────────────────────────────────────────────────────────────
# Trigger
# ─────────────────────────────────────────────────────────────────────────────

def test_trigger_event_count():
    trig = _make_trigger(n_events=10)
    idx, _, times = trig.get_events()
    assert len(idx) == 10


def test_trigger_event_times_positive():
    trig = _make_trigger()
    _, _, times = trig.get_events()
    assert np.all(times >= 0)


def test_trigger_normalized_binary():
    trig = _make_trigger()
    norm = trig.normalized
    assert set(np.unique(norm)).issubset({0, 1})


def test_trigger_shape_mismatch_raises():
    n = int(DURATION * FS)
    with pytest.raises(ValueError):
        Trigger(data=np.zeros(n), t=np.arange(n + 5) / FS)
