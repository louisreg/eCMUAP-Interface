"""
Tests for eCMUAPView.reject_epochs() and the updated average(reject=True).
"""

import numpy as np
import pytest

from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.utils.trigger import Trigger
from ecmuap_interface.views.eCMUAP_view import eCMUAPView


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

FS          = 10_000.0
T_PRE       = 5e-3
T_POST      = 20e-3
N_EVENTS    = 20
N_CHANNELS  = 4


def _make_view(n_artifact=2, artifact_scale=10.0, seed=42):
    """
    Build a synthetic eCMUAPView with *n_artifact* epochs that are
    *artifact_scale* × louder than the clean epochs.
    """
    rng = np.random.default_rng(seed)

    # Trigger: one event every 100 ms
    t_event_ms    = np.arange(N_EVENTS) * 100.0 + 50.0   # ms
    t_event_s     = t_event_ms * 1e-3

    n_pre  = int(round(T_PRE  * FS))
    n_post = int(round(T_POST * FS))
    epoch_len = n_pre + n_post

    total_samples = int(t_event_s[-1] * FS) + n_post + 500
    data = rng.normal(0, 1.0, (N_CHANNELS, total_samples))
    time = np.arange(total_samples) / FS

    # Inject artifact into the first n_artifact events
    artifact_idx = [int(round(te * FS)) for te in t_event_s[:n_artifact]]
    for idx in artifact_idx:
        s, e = idx - n_pre, idx + n_post
        data[:, s:e] += artifact_scale * rng.normal(0, 1.0, (N_CHANNELS, epoch_len))

    # Build trigger from event times
    trig_data = np.zeros(total_samples)
    for te in t_event_s:
        i = int(round(te * FS))
        if i < total_samples:
            trig_data[i] = 5.0  # above threshold=2

    trigger = Trigger(data=trig_data, t=time)
    emg     = EMGData(data=data, time=time, trigger=trigger)
    return eCMUAPView(emg)


# ─────────────────────────────────────────────────────────────────────────────
# Basic shape / return type
# ─────────────────────────────────────────────────────────────────────────────

def test_reject_returns_array():
    view = _make_view(n_artifact=0)
    clean = view.reject_epochs(T_PRE, T_POST)
    assert isinstance(clean, np.ndarray)
    assert clean.ndim == 3   # (n_epochs, n_channels, n_samples)


def test_reject_returns_mask_when_requested():
    view = _make_view(n_artifact=2)
    clean, mask = view.reject_epochs(T_PRE, T_POST, return_mask=True)
    assert mask.dtype == bool
    assert mask.shape == (N_EVENTS,)
    assert clean.shape[0] == mask.sum()


def test_no_artifact_all_accepted():
    view  = _make_view(n_artifact=0)
    _, mask = view.reject_epochs(T_PRE, T_POST, return_mask=True)
    assert mask.all(), "No artefacts injected — all epochs should be accepted"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection actually removes artefact epochs
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["amplitude", "rms"])
def test_artifacts_rejected(method):
    n_art = 3
    view  = _make_view(n_artifact=n_art, artifact_scale=20.0)
    clean, mask = view.reject_epochs(
        T_PRE, T_POST,
        method=method,
        return_mask=True,
    )
    n_rejected = (~mask).sum()
    assert n_rejected >= n_art, \
        f"{method}: expected ≥{n_art} rejections, got {n_rejected}"


def _make_shape_artifact_view(n_artifact=3, seed=42):
    """
    Clean epochs contain a consistent sinusoidal evoked response (amplitude ~1).
    Artifact epochs have the *same* amplitude but sign-flipped shape — so
    amplitude / RMS thresholds cannot detect them, but correlation can.
    """
    rng = np.random.default_rng(seed)

    t_event_ms  = np.arange(N_EVENTS) * 100.0 + 50.0
    t_event_s   = t_event_ms * 1e-3

    n_pre     = int(round(T_PRE  * FS))
    n_post    = int(round(T_POST * FS))
    epoch_len = n_pre + n_post

    total_samples = int(t_event_s[-1] * FS) + n_post + 500
    time = np.arange(total_samples) / FS
    data = rng.normal(0, 0.05, (N_CHANNELS, total_samples))   # tiny noise floor

    t_ep   = np.arange(epoch_len) / FS
    evoked = np.sin(2 * np.pi * 50.0 * t_ep) * np.exp(-t_ep / (T_POST / 2))

    # All events get the clean evoked response
    for te in t_event_s:
        i = int(round(te * FS))
        s, e = i - n_pre, i + n_post
        if s >= 0 and e <= total_samples:
            data[:, s:e] += evoked[np.newaxis, :]

    # Artifact epochs: flip sign (same amplitude, orthogonal shape)
    artifact_idx = [int(round(te * FS)) for te in t_event_s[:n_artifact]]
    for idx in artifact_idx:
        s, e = idx - n_pre, idx + n_post
        if s >= 0 and e <= total_samples:
            data[:, s:e] = -data[:, s:e]   # sign flip preserves amplitude

    trig_data = np.zeros(total_samples)
    for te in t_event_s:
        i = int(round(te * FS))
        if i < total_samples:
            trig_data[i] = 5.0

    trigger = Trigger(data=trig_data, t=time)
    emg     = EMGData(data=data, time=time, trigger=trigger)
    return eCMUAPView(emg)


def test_correlation_method_rejects_artifacts():
    """Correlation rejection detects shape-different (sign-flipped) epochs
    that have the same amplitude as clean epochs — amplitude/RMS cannot
    detect these, but correlation can."""
    n_art = 3
    view  = _make_shape_artifact_view(n_artifact=n_art)

    # Confirm amplitude method does NOT catch the flipped epochs
    _, mask_amp = view.reject_epochs(
        T_PRE, T_POST,
        method="amplitude",
        return_mask=True,
    )
    assert mask_amp.all(), "amplitude should keep all epochs (same amplitude)"

    # Correlation method SHOULD catch them
    clean, mask = view.reject_epochs(
        T_PRE, T_POST,
        method="correlation",
        return_mask=True,
    )
    n_rejected = (~mask).sum()
    assert n_rejected >= n_art, \
        f"correlation: expected ≥{n_art} rejections, got {n_rejected}"


# ─────────────────────────────────────────────────────────────────────────────
# Manual threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_very_high_threshold_accepts_all():
    view = _make_view(n_artifact=2, artifact_scale=5.0)
    _, mask = view.reject_epochs(
        T_PRE, T_POST,
        method="amplitude",
        threshold=1e9,
        return_mask=True,
    )
    assert mask.all()


def test_very_low_threshold_rejects_all():
    view = _make_view(n_artifact=0)
    _, mask = view.reject_epochs(
        T_PRE, T_POST,
        method="amplitude",
        threshold=0.0,
        return_mask=True,
    )
    assert not mask.any()


# ─────────────────────────────────────────────────────────────────────────────
# average(reject=True) integration
# ─────────────────────────────────────────────────────────────────────────────

def test_average_with_rejection_shape():
    view = _make_view(n_artifact=2)
    avg  = view.average(T_PRE, T_POST, reject=True)
    n_ep = int(round((T_PRE + T_POST) * FS))
    assert avg.shape == (N_CHANNELS, n_ep)


def test_average_reject_reduces_noise():
    """Averaging with artifact rejection should produce a lower-amplitude result
    than averaging with the artifacts included (which inflate the mean)."""
    view      = _make_view(n_artifact=4, artifact_scale=50.0)
    avg_clean = view.average(T_PRE, T_POST, reject=True)
    avg_all   = view.average(T_PRE, T_POST, reject=False)
    rms_clean = float(np.sqrt(np.mean(avg_clean ** 2)))
    rms_all   = float(np.sqrt(np.mean(avg_all ** 2)))
    assert rms_clean < rms_all, \
        "Rejection should lower the average RMS when artefacts are present"


# ─────────────────────────────────────────────────────────────────────────────
# Invalid method raises
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_method_raises():
    view = _make_view()
    with pytest.raises(ValueError, match="Unknown rejection method"):
        view.reject_epochs(T_PRE, T_POST, method="unknown")
