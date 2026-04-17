"""
Pandas export utilities for eCMUAP metrics.

Main entry point
----------------
:func:`to_dataframe` — collect per-electrode eCMUAP metrics (amplitude,
timing, propagation velocity) from an ``HDEMGView`` into a tidy
``pandas.DataFrame``, ready for downstream statistics (R, SPSS, seaborn …).

Typical workflow
----------------
::

    from ecmuap_interface.utils.export import to_dataframe

    df = to_dataframe(hd_view, t_pre=1e-3, t_post=15e-3,
                      filter_label="IB2", include_velocity=True)
    df.to_csv("results.csv", index=False)

    # Compare across filters
    dfs = [to_dataframe(v, 1e-3, 15e-3, filter_label=lbl)
           for lbl, v in labeled_views]
    all_df = pd.concat(dfs, ignore_index=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecmuap_interface.views.eCMUAP_view import eCMUAPView
from ecmuap_interface.core.eCMUAP import eCMUAP


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def to_dataframe(
    hd_view,
    t_pre: float,
    t_post: float,
    skip_start: int = 0,
    skip_end: int = 0,
    filter_label: str = "",
    reject: bool = False,
    reject_method: str = "amplitude",
    reject_threshold: float | None = None,
    reject_auto_scale: float = 3.0,
    include_velocity: bool = False,
) -> pd.DataFrame:
    """
    Collect per-electrode eCMUAP metrics into a tidy ``pandas.DataFrame``.

    Parameters
    ----------
    hd_view : HDEMGView
        View object wrapping an :class:`~ecmuap_interface.core.HD_emg.HDEMG`.
    t_pre, t_post : float
        Epoch window around the trigger (s).
    skip_start, skip_end : int
        Trigger events to discard at the edges.
    filter_label : str
        Label stored in the ``filter`` column (e.g. ``"IB2"``).
        Use an empty string for the unfiltered reference.
    reject : bool
        Apply automatic epoch rejection before averaging.
    reject_method, reject_threshold, reject_auto_scale
        Passed to
        :meth:`~ecmuap_interface.views.eCMUAP_view.eCMUAPView.reject_epochs`.
    include_velocity : bool
        If True **and** the probe is a uniform grid, add ``delay_ms`` and
        ``speed_ms`` columns (from the delay-map / gradient method).
        Requires the ``pitch_um`` probe annotation.

    Returns
    -------
    pd.DataFrame
        One row per electrode.  Columns:

        ==================  =============================================
        electrode_id        0-based channel index
        x_um, y_um          electrode position (µm)
        rms_uV              root-mean-square amplitude (µV)
        ptp_uV              peak-to-peak amplitude (µV)
        peak_uV             maximum absolute amplitude (µV)
        latency_ms          eCMUAP onset latency (ms, 10 % threshold)
        duration_ms         eCMUAP duration (ms)
        ttmin_ms            time of negative peak (ms relative to trigger)
        ttmax_ms            time of positive peak (ms)
        delay_ms            [velocity] absolute time of negative peak (ms)
        speed_ms            [velocity] local propagation speed (m/s)
        filter              ``filter_label`` argument
        t_pre_s             epoch window parameter
        t_post_s            epoch window parameter
        ==================  =============================================

        NaN for electrodes outside the probe convex hull (spatial
        interpolation artefacts) or with insufficient signal.
    """
    from ecmuap_interface.views.HDemg_view import HDEMGView  # avoid circular

    if not isinstance(hd_view, HDEMGView):
        raise TypeError("hd_view must be an HDEMGView instance.")

    emg   = hd_view.emg
    probe = hd_view.probe
    pos   = probe.contact_positions
    n_ch  = emg.n_channels

    # ── Average eCMUAP ───────────────────────────────────────────────────────
    cmuap_view = eCMUAPView(emg)
    avg = cmuap_view.average(
        t_pre, t_post,
        skip_start=skip_start, skip_end=skip_end,
        reject=reject,
        reject_method=reject_method,
        reject_threshold=reject_threshold,
        reject_auto_scale=reject_auto_scale,
    )

    epoch_time = np.linspace(-t_pre, t_post, avg.shape[1], endpoint=False)

    # ── Per-electrode amplitude / timing metrics ──────────────────────────────
    _METRICS = ("rms", "peak2peak", "peak", "latency", "duration", "ttmin", "ttmax")
    records: list[dict] = []

    for ch in range(n_ch):
        sig = avg[ch]
        row: dict = {
            "electrode_id": ch,
            "x_um":         float(pos[ch, 0]),
            "y_um":         float(pos[ch, 1]),
        }

        if np.any(np.isfinite(sig)):
            ec = eCMUAP(sig, epoch_time)
            row["rms_uV"]      = ec.rms
            row["ptp_uV"]      = ec.peak2peak
            row["peak_uV"]     = ec.peak
            row["latency_ms"]  = ec.latency  * 1e3
            row["duration_ms"] = ec.duration * 1e3
            row["ttmin_ms"]    = ec.ttmin    * 1e3
            row["ttmax_ms"]    = ec.ttmax    * 1e3
        else:
            for key in ("rms_uV", "ptp_uV", "peak_uV",
                        "latency_ms", "duration_ms", "ttmin_ms", "ttmax_ms"):
                row[key] = np.nan

        records.append(row)

    df = pd.DataFrame(records)

    # ── Optional velocity columns ─────────────────────────────────────────────
    if include_velocity:
        df = _add_velocity_columns(df, avg, emg.fs, probe)

    # ── Metadata columns ──────────────────────────────────────────────────────
    df["filter"]  = filter_label
    df["t_pre_s"] = t_pre
    df["t_post_s"] = t_post

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Private helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_velocity_columns(
    df: pd.DataFrame,
    avg: np.ndarray,
    fs: float,
    probe,
) -> pd.DataFrame:
    """Append ``delay_ms`` and ``speed_ms`` columns using the delay-gradient method."""
    from ecmuap_interface.utils.probes import (
        is_uniform_grid,
        get_grid_indices_from_probe,
        get_grid_shape_from_probe,
    )
    from ecmuap_interface.utils.velocity import (
        compute_delay_map,
        compute_velocity_map,
    )

    df = df.copy()
    df["delay_ms"] = np.nan
    df["speed_ms"] = np.nan

    if not is_uniform_grid(probe):
        return df   # velocity map requires a uniform grid

    pitch = float(probe.annotations.get("pitch_um", 0))
    if pitch == 0:
        return df

    delays = compute_delay_map(avg, fs)   # (n_channels,)
    df["delay_ms"] = delays

    # Reshape delays to grid for gradient-based speed
    n_rows, n_cols = get_grid_shape_from_probe(probe)
    delay_grid     = np.full((n_rows, n_cols), np.nan)
    grid_idx       = get_grid_indices_from_probe(probe)
    for ch, (r, c) in grid_idx.items():
        delay_grid[r, c] = delays[ch]

    speed_grid, _, _ = compute_velocity_map(delay_grid, pitch)

    # Map grid → flat channel order
    for ch, (r, c) in grid_idx.items():
        df.loc[df["electrode_id"] == ch, "speed_ms"] = float(speed_grid[r, c])

    return df
