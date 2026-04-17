import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List

from ..core.eCMUAP import eCMUAP
from ..core.emg_data import EMGData


class eCMUAPView:
    """
    eCMUAP processing and visualization view.

    Works for mono- or multi-channel EMG without special casing.
    """

    def __init__(self, emg: EMGData):
        self.emg = emg
        self.trigger = emg.trigger

        if self.trigger is None:
            raise ValueError("EMGData has no trigger associated")

    # ==================================================
    # EPOCHING
    # ==================================================
    def epochs(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
    ) -> NDArray:
        """
        Extract stimulus-aligned epochs.

        Parameters
        ----------
        t_pre : float
            Time before trigger (s, positive)
        t_post : float
            Time after trigger (s, positive)
        skip_start : int
            Number of initial trigger events to skip
        skip_end : int
            Number of final trigger events to skip

        Returns
        -------
        epochs : NDArray
            Shape (n_events, n_channels, n_samples_epoch)
        """

        n_pre = int(round(t_pre * self.emg.fs))
        n_post = int(round(t_post * self.emg.fs))

        event_idx_all = self.trigger.get_events()[0]

        if skip_start < 0 or skip_end < 0:
            raise ValueError("skip_start and skip_end must be >= 0")

        if skip_start + skip_end >= len(event_idx_all):
            raise ValueError("skip_start + skip_end removes all trigger events")

        event_idx = event_idx_all[
            skip_start : len(event_idx_all) - skip_end
        ]

        epochs = []
        n_skipped = 0

        for idx in event_idx:
            start = idx - n_pre
            stop = idx + n_post

            if start < 0 or stop > self.emg.n_samples:
                n_skipped += 1
                continue

            ep = self.emg.data[:, start:stop]
            epochs.append(ep)

        if n_skipped > 0:
            warnings.warn(
                f"{n_skipped} epoch(s) skipped: window extends outside signal bounds "
                f"(t_pre={t_pre}s, t_post={t_post}s, signal_duration="
                f"{self.emg.n_samples / self.emg.fs:.3f}s).",
                UserWarning,
                stacklevel=2,
            )

        if len(epochs) == 0:
            raise RuntimeError("No valid epochs extracted")

        return np.stack(epochs, axis=0)

    # ==================================================
    # EPOCH REJECTION
    # ==================================================
    def reject_epochs(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        method: str = "amplitude",
        threshold: float | None = None,
        auto_scale: float = 3.0,
        return_mask: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Reject artefact-contaminated epochs before averaging.

        Designed for evoked HD-EMG: a single noisy trial (movement artefact,
        failed stimulus, saturated amplifier) can distort the whole average.

        Parameters
        ----------
        t_pre, t_post : float
            Epoch window (s).
        skip_start, skip_end : int
            Events to discard at the edges (passed to :meth:`epochs`).
        method : {"amplitude", "rms", "correlation"}
            Rejection criterion:

            * ``"amplitude"``   — reject if ``max(|signal|)`` across all
              channels exceeds *threshold*.  Fast and robust for artefacts.
            * ``"rms"``         — reject if RMS across all channels exceeds
              *threshold*.
            * ``"correlation"`` — reject if correlation with the trial-average
              template falls below *threshold* (lower = more aggressive).
        threshold : float or None
            Rejection threshold in the same units as the criterion score.
            If ``None``, set automatically as ``auto_scale × median(scores)``.
        auto_scale : float
            Multiplier for auto threshold.  Default 3.0 (reject >3× median).
            For ``"correlation"`` method this is the *maximum tolerated*
            deviation: threshold = ``1 - 1/auto_scale``.
        return_mask : bool
            If True, also return the boolean acceptance mask.

        Returns
        -------
        clean_epochs : NDArray, shape (n_accepted, n_channels, n_samples)
        accept_mask  : NDArray, shape (n_total,)   — only when return_mask=True
        """
        epochs = self.epochs(t_pre, t_post, skip_start, skip_end)
        n_epochs = epochs.shape[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if method == "amplitude":
                scores = np.nanmax(np.abs(epochs), axis=(1, 2))
            elif method == "rms":
                scores = np.sqrt(np.nanmean(epochs ** 2, axis=(1, 2)))
            elif method == "correlation":
                if n_epochs == 0 or not np.any(np.isfinite(epochs)):
                    scores = np.full(n_epochs, np.nan)
                else:
                    template = np.nanmean(epochs, axis=0).ravel()
                    scores = np.array([
                        1.0 - float(np.corrcoef(epochs[i].ravel(), template)[0, 1])
                        for i in range(n_epochs)
                    ])
            else:
                raise ValueError(
                    f"Unknown rejection method {method!r}. "
                    "Choose 'amplitude', 'rms', or 'correlation'."
                )

        if threshold is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = float(np.nanmedian(scores))
            if not np.isfinite(med):
                # All scores are NaN — nothing to reject
                accept_mask = np.ones(n_epochs, dtype=bool)
                if return_mask:
                    return epochs, accept_mask
                return epochs
            if method == "correlation":
                # Median + auto_scale × MAD — robust to near-zero medians
                mad = float(np.nanmedian(np.abs(scores - med)))
                mad = mad if np.isfinite(mad) and mad > 0 else 0.0
                threshold = med + auto_scale * mad
            else:
                threshold = auto_scale * med

        accept_mask = scores <= threshold
        n_rejected  = int((~accept_mask).sum())

        if n_rejected > 0:
            warnings.warn(
                f"reject_epochs ({method}): {n_rejected}/{n_epochs} epoch(s) "
                f"rejected (threshold={threshold:.4g}).",
                UserWarning,
                stacklevel=2,
            )

        clean_epochs = epochs[accept_mask]

        if return_mask:
            return clean_epochs, accept_mask
        return clean_epochs

    # ==================================================
    # AVERAGING
    # ==================================================
    def average(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        reject: bool = False,
        reject_method: str = "amplitude",
        reject_threshold: float | None = None,
        reject_auto_scale: float = 3.0,
    ) -> NDArray[np.float64]:
        """
        Average eCMUAP across events.

        Parameters
        ----------
        reject : bool
            If True, apply automatic epoch rejection before averaging.
        reject_method : str
            Passed to :meth:`reject_epochs`.
        reject_threshold : float or None
            Rejection threshold (auto if None).
        reject_auto_scale : float
            Auto-threshold multiplier (default 3.0).

        Returns
        -------
        avg : NDArray, shape (n_channels, n_samples_epoch)
        """
        if reject:
            epochs = self.reject_epochs(
                t_pre, t_post,
                skip_start=skip_start, skip_end=skip_end,
                method=reject_method,
                threshold=reject_threshold,
                auto_scale=reject_auto_scale,
            )
        else:
            epochs = self.epochs(t_pre, t_post,
                                 skip_start=skip_start, skip_end=skip_end)
        return np.mean(epochs, axis=0)

    def epoch_time(self, t_pre: float, t_post: float, n_samples: int) -> NDArray:
        """
        Time vector for one epoch.
        """
        return np.linspace(-t_pre, t_post, n_samples, endpoint=False)

    # ==================================================
    # eCMAP HELPERS
    # ==================================================
    def average_ecmap(
        self,
        ch_idx: int,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
    ) -> eCMUAP:
        """
        Return averaged eCMAP object for one channel.
        """
        avg = self.average(
            t_pre,
            t_post,
            skip_start=skip_start,
            skip_end=skip_end,
        )[ch_idx]

        time = self.epoch_time(
            t_pre,
            t_post,
            avg.shape[0],
        )

        return eCMUAP(avg, time)

    # ==================================================
    # PLOTTING
    # ==================================================
    def plot_average(
        self,
        ch_idx: int,
        t_pre: float,
        t_post: float,
        ax: plt.Axes | None = None,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Plot averaged eCMUAP for one channel.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))

        ecmap = self.average_ecmap(ch_idx, t_pre, t_post)

        ax.plot(ecmap.t, ecmap.data, **plot_kwargs)
        ax.axvline(0.0, color="k", linestyle="--", alpha=0.6)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG (µV)")
        ax.set_title(f"Average eCMUAP – channel {ch_idx}")

        return ax

    def plot_metrics_markers(
        self,
        ch_idx: int,
        t_pre: float,
        t_post: float,
        metrics: list[str] | None = None,
        ax: plt.Axes | None = None,
        color: str = "C1",
    ) -> plt.Axes:
        """
        Overlay metric markers on averaged eCMUAP plot.

        Supported metrics:
            - latency
            - ttmin
            - ttmax
            - min
            - max
        """

        if ax is None:
            ax = self.plot_average(ch_idx, t_pre, t_post)

        ecmap = self.average_ecmap(ch_idx, t_pre, t_post)
        summary = ecmap.summary().iloc[0]

        # default metrics
        metrics = metrics or ["latency", "ttmin", "ttmax", "min", "max"]

        for m in metrics:
            if m not in summary:
                continue

            # --------------------------------------------------
            # TIME-BASED METRICS
            # --------------------------------------------------
            if m in ["latency", "ttmin", "ttmax"]:
                t_val = summary[m]

                ax.axvline(
                    t_val,
                    color=color,
                    linestyle=":",
                    alpha=0.8,
                    label=m,
                )

                idx = int(np.argmin(np.abs(ecmap.t - t_val)))
                ax.plot(
                    ecmap.t[idx],
                    ecmap.data[idx],
                    "o",
                    color=color,
                )

            # --------------------------------------------------
            # AMPLITUDE-BASED METRICS
            # --------------------------------------------------
            elif m in ["min", "max"]:
                if m == "min":
                    idx = ecmap.min_idx
                else:
                    idx = ecmap.max_idx

                t_val = ecmap.t[idx]
                y_val = ecmap.data[idx]

                ax.axvline(
                    t_val,
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    label=m,
                )

                ax.plot(
                    t_val,
                    y_val,
                    "s",
                    color=color,
                )

        ax.legend()
        return ax
