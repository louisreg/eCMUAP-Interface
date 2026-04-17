import numpy as np
from numpy.typing import NDArray
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from ecmuap_interface.utils import filters
from ecmuap_interface.utils.functions import smooth_start_end

class eCMUAP:
    """
    Evoked Compound Muscle Action Potential (single channel, single epoch).
    """

    def __init__(self, data: NDArray, t: NDArray):
        if data.ndim != 1:
            raise ValueError("eCMAP data must be 1D (single channel)")

        if data.size != t.size:
            raise ValueError("data and time must have the same length")

        self._raw = np.asarray(data, dtype=float)
        self._data = self._raw.copy()
        self._t = np.asarray(t, dtype=float)

        self.n_samples = self._data.size
        dt = np.mean(np.diff(self._t))
        self.fs = 1.0 / dt

    # ==================================================
    # BASIC PROPERTIES
    # ==================================================
    @property
    def t(self) -> NDArray:
        return self._t

    @property
    def raw(self) -> NDArray:
        return self._raw

    @property
    def data(self) -> NDArray:
        return self._data

    # ==================================================
    # AMPLITUDE METRICS
    # ==================================================
    @property
    def min(self) -> float:
        return float(np.min(self._data))

    @property
    def max(self) -> float:
        return float(np.max(self._data))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self._data)))

    @property
    def peak2peak(self) -> float:
        return float(self.max - self.min)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self._data**2)))

    # ==================================================
    # TIME METRICS
    # ==================================================
    @property
    def min_idx(self) -> int:
        return int(np.argmin(self._data))

    @property
    def max_idx(self) -> int:
        return int(np.argmax(self._data))

    @property
    def peak_idx(self) -> int:
        return int(np.argmax(np.abs(self._data)))

    @property
    def ttmin(self) -> float:
        return float(self._t[self.min_idx])

    @property
    def ttmax(self) -> float:
        return float(self._t[self.max_idx])

    @property
    def ttpeak(self) -> float:
        return float(self._t[self.peak_idx])

    # ==================================================
    # LATENCY / DURATION (10% threshold)
    # ==================================================
    def _rectified(self) -> NDArray:
        return np.abs(self._data)

    def _threshold_indices(self, frac: float = 0.1) -> NDArray:
        thr = frac * self.peak
        return np.where(self._rectified() >= thr)[0]

    @property
    def tmin_10(self) -> float:
        idxs = self._threshold_indices()
        return float(self._t[idxs[0]])

    @property
    def tmax_10(self) -> float:
        idxs = self._threshold_indices()
        return float(self._t[idxs[-1]])

    @property
    def latency(self) -> float:
        return self.tmin_10

    @property
    def duration(self) -> float:
        return self.tmax_10 - self.tmin_10

    # ==================================================
    # SIGNAL PROCESSING (IN-PLACE)
    # ==================================================
    def truncate(self, tmax: float) -> None:
        mask = self._t <= tmax
        self._t = self._t[mask]
        self._data = self._data[mask]
        self._raw = self._raw[mask]
        self.n_samples = self._data.size

    def apply_filter(self, filt: filters.Filter) -> NDArray:
        self._data = filt(self._data, self.fs)
        return self._data

    def HPF(self, cutoff: float, order: int = 5) -> NDArray:
        filt = filters.butter_HPF(cutoff, order)
        self._data = filt(self._data, self.fs)
        return self._data

    def LPF(self, cutoff: float, order: int = 5) -> NDArray:
        filt = filters.butter_LPF(cutoff, order)
        self._data = filt(self._data, self.fs)
        return self._data

    def notch(self, freqs=60.0, Q=30.0) -> None:
        filt = filters.IIRNotchFilter(freqs, Q)
        self._data = filt(self._data, self.fs)

    def detrend(self, frac: float = 0.4, polyorder: int = 3) -> None:
        self._data = signal.detrend(self._data, type="linear")

        if frac <= 0:
            return

        N = self._data.size
        win = max(int(N * frac), polyorder + 2)
        if win % 2 == 0:
            win += 1
        win = min(win, N if N % 2 else N - 1)

        trend = signal.savgol_filter(self._data, win, polyorder)
        self._data -= trend

    def denoise(self, mysize: int = 9) -> None:
        self._data = signal.wiener(self._data, mysize=mysize)

    def smooth_start_end(self, n_start: int = 50, n_end: int = 150) -> None:
        self._data = smooth_start_end(self._data, n_start, n_end)

    # ==================================================
    # SPECTRAL
    # ==================================================
    def PSD(self, raw: bool = False, **welch_kwargs):
        x = self._raw if raw else self._data
        return signal.welch(x, fs=self.fs, **welch_kwargs)

    # ==================================================
    # PLOTTING
    # ==================================================
    def plot(self, ax: plt.Axes, raw: bool = False, **kwargs):
        y = self._raw if raw else self._data
        ax.plot(self._t, y, **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG (µV)")
        ax.set_xlim(self._t[0], self._t[-1])

    # ==================================================
    # SUMMARY
    # ==================================================
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "max": [self.max],
                "min": [self.min],
                "peak2peak": [self.peak2peak],
                "rms": [self.rms],
                "ttmax": [self.ttmax],
                "ttmin": [self.ttmin],
                "latency": [self.latency],
                "duration": [self.duration],
            }
        )
