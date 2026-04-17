import numpy as np
from scipy import signal
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple


class Trigger:
    """
    Trigger signal handler.

    This class handles a digital or analog trigger signal and provides:
    - normalization to binary (0 / 1)
    - detection of trigger onset events
    - inter-event sample segmentation
    - plotting utilities
    """

    def __init__(self, data: NDArray, t: NDArray):
        """
        Parameters
        ----------
        data : NDArray
            Trigger signal (analog or digital)
        t : NDArray
            Time vector (seconds), same length as data
        """
        self._data = np.asarray(data)
        self._t = np.asarray(t)

        if self._data.shape != self._t.shape:
            raise ValueError("data and t must have the same shape")

        self._n_samples = self._data.size
        self._normalized: NDArray | None = None

    # ==================================================
    # BASIC PROPERTIES
    # ==================================================
    @property
    def t(self) -> NDArray:
        """Time vector (seconds)."""
        return self._t

    @property
    def raw(self) -> NDArray:
        """Raw trigger signal."""
        return self._data

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._n_samples

    # ==================================================
    # NORMALIZATION
    # ==================================================
    def _normalize(self) -> NDArray:
        """
        Normalize trigger signal to binary (0 or 1).

        Any value > 2 is considered a trigger (1),
        everything else is set to 0.

        Returns
        -------
        NDArray
            Binary trigger signal
        """
        if self._normalized is None:
            norm = np.zeros_like(self._data, dtype=int)
            norm[self._data > 2] = 1
            self._normalized = norm

        return self._normalized

    def skip_event(self, event_id: int) -> int:
        """
        Remove all samples before a given trigger event.
        """

        event_idx, _, _ = self.get_events()

        if event_id < 0 or event_id >= len(event_idx):
            raise IndexError("Invalid event_id")

        shift = int(event_idx[event_id])

        if shift <= 0:
            return 0

        # use private attributes directly
        self._data = self._data[shift:]
        self._t = self._t[shift:] - self._t[shift]

        # reset cache
        self._normalized = None

        return shift

    @property
    def normalized(self) -> NDArray:
        """Binary (0/1) trigger signal."""
        return self._normalize()

    # ==================================================
    # EVENT DETECTION
    # ==================================================
    def get_events(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Detect trigger onset events.

        Returns
        -------
        event_idx : NDArray
            Sample indices of trigger onsets
        event_values : NDArray
            Trigger values at onsets (should be all ones)
        event_times : NDArray
            Event times in seconds
        """
        # Find rising edges: diff goes from 0 -> 1
        trig = self._normalize()
        rising_edges = np.where(np.diff(trig, prepend=0) == 1)[0]

        return (
            rising_edges,
            trig[rising_edges],
            self._t[rising_edges],
        )

    # ==================================================
    # INTER-EVENT SEGMENTATION
    # ==================================================
    def get_inter_event_samples(self) -> list[NDArray]:
        """
        Get sample indices between successive trigger events.

        Returns
        -------
        list[NDArray]
            List of index arrays corresponding to inter-event segments
        """
        event_idx, _, _ = self.get_events()

        segments = []
        for start, stop in zip(event_idx[:-1], event_idx[1:]):
            segments.append(np.arange(start, stop))

        # Last segment: from last event to end
        segments.append(np.arange(event_idx[-1], self._n_samples))

        return segments

    # ==================================================
    # PLOTTING
    # ==================================================
    def plot_raw(self, ax: plt.Axes, **kwargs):
        """Plot raw trigger signal."""
        ax.plot(self._t, self._data, **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trigger (raw)")
        ax.set_xlim(self._t[0], self._t[-1])

    def plot_normalized(self, ax: plt.Axes, **kwargs):
        """Plot normalized trigger signal."""
        ax.plot(self._t, self._normalize(), **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trigger (binary)")
        ax.set_xlim(self._t[0], self._t[-1])
