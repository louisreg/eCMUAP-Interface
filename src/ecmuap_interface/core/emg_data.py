from numpy.typing import NDArray
from scipy.signal import welch
import numpy as np
from ..utils import filters
from ..utils.trigger import Trigger

from ..utils.spatial_filters import kernel


class EMGData:
    """
    Continuous EMG recording.

    data.shape = (n_channels, n_samples)
    """

    def __init__(
        self,
        data: NDArray,
        time: NDArray,
        trigger: Trigger | None = None,
        spatial_filter: kernel | None = None,
    ):
        if data.ndim != 2:
            raise ValueError("data must be (n_channels, n_samples)")

        self.data = np.asarray(data, dtype=float)
        self.raw = self.data.copy()
        self.time = np.asarray(time, dtype=float)

        self.n_channels, self.n_samples = self.data.shape
        self.fs = 1.0 / np.mean(np.diff(self.time))
        self.trigger = trigger

        self.spatial_filter = spatial_filter

    # ==================================================
    # ACCESSORS
    # ==================================================
    def channel(self, ch_idx: int) -> NDArray:
        """Return one channel (view)."""
        return self.data[ch_idx]

    # ==================================================
    # FILTERING (IN-PLACE)
    # ==================================================
    def apply_filter(self, filt: filters.Filter) -> None:
        self.data = filt(self.data, self.fs)

    def HPF(self, cutoff: float, order: int = 5) -> None:
        filt = filters.butter_HPF(cutoff, order)
        self.data = filt(self.data, self.fs)

    def LPF(self, cutoff: float, order: int = 5) -> None:
        filt = filters.butter_LPF(cutoff, order)
        self.data = filt(self.data, self.fs)

    def notch(self, freqs=50.0, Q=30.0) -> None:
        filt = filters.IIRNotchFilter(freqs, Q)
        self.data = filt(self.data, self.fs)

    def reset(self) -> None:
        """Restore raw signal."""
        self.data = self.raw.copy()

    # ==================================================
    # SPECTRAL
    # ==================================================
    def PSD(self, ch_idx: int, raw: bool = False, **welch_kwargs):
        x = self.raw[ch_idx] if raw else self.data[ch_idx]
        return welch(x, fs=self.fs, **welch_kwargs)


    def remove_baseline(
        self,
        factor: float = 3.0,
        min_duration: float = 0.5,
    ) -> bool:
        """
        Detect an abnormally large delay between the first two trigger events.
        If detected:
          1) compute the mean signal between trigger 0 and trigger 1
          2) subtract it from the whole signal
          3) shift the signal in time so that trigger 1 becomes the first event

        Parameters
        ----------
        factor : float
            Threshold factor relative to the median inter-event interval
        min_duration : float
            Minimum duration (s) to consider the gap as valid baseline

        Returns
        -------
        applied : bool
            True if correction was applied
        """

        if self.trigger is None:
            raise ValueError("No trigger associated with EMGData")

        # Trigger event indices and times
        event_idx, _, event_t = self.trigger.get_events()

        if len(event_idx) < 2:
            return False

        # Inter-event intervals (s)
        dt = np.diff(event_t)

        # Reference interval (robust)
        ref_dt = np.median(dt[1:]) if len(dt) > 1 else dt[0]

        # Check abnormal initial gap
        if dt[0] < factor * ref_dt:
            return False

        if dt[0] < min_duration:
            return False

        # ------------------------------------------
        # Compute baseline from initial gap
        # ------------------------------------------
        idx_start = event_idx[0]
        idx_stop = event_idx[1]

        baseline = np.mean(
            self.data[:, idx_start:idx_stop],
            axis=1,
            keepdims=True,
        )

        # Subtract baseline
        self.data = self.data - baseline

        # ------------------------------------------
        # Shift signal so trigger 1 becomes trigger 0
        # ------------------------------------------
        shift = idx_stop

        self.data = self.data[:, shift:]
        self.raw = self.raw[:, shift:]
        self.time = self.time[shift:] - self.time[shift]

        self.n_samples = self.data.shape[1]

        # ------------------------------------------
        # Shift trigger consistently
        # ------------------------------------------
        self.trigger.skip_event(1)


        return True

    # ==================================================
    # METRICS
    # ==================================================
    def rms(
        self,
        raw: bool = False,
    ) -> NDArray:
        """
        Compute RMS value for each channel.

        Parameters
        ----------
        raw : bool
            If True, compute RMS on raw signal.
            If False, compute RMS on filtered signal.

        Returns
        -------
        NDArray
            RMS values, shape = (n_channels,)
        """
        x = self.raw if raw else self.data
        return np.sqrt(np.mean(x ** 2, axis=1))