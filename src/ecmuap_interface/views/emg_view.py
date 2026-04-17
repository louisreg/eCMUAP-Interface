import matplotlib.pyplot as plt
import numpy as np

from ..core.emg_data import EMGData


class EMGView:
    """
    Visualization utilities for continuous EMG.
    """

    def __init__(self, emg: EMGData):
        self.emg = emg

    def plot(
        self,
        channels: list[int] | None = None,
        tlim: tuple[float, float] | None = None,
        offset: float | None = None,
        raw: bool = False,
        show_triggers: bool = True,
        trigger_kwargs: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot continuous EMG signals.

        Parameters
        ----------
        channels : list[int] | None
            Channels to plot (default: all)
        tlim : (tmin, tmax) | None
            Time window (s)
        offset : float | None
            Vertical offset between channels
        raw : bool
            Plot raw instead of filtered data
        show_triggers : bool
            Overlay trigger events if available
        trigger_kwargs : dict | None
            Keyword arguments passed to ax.axvline for triggers
        ax : matplotlib Axes | None
            Existing axes
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        if channels is None:
            channels = list(range(self.emg.n_channels))

        data = self.emg.raw if raw else self.emg.data

        if offset is None:
            offset = 2 * np.std(data)

        # --------------------------------------------------
        # Plot EMG channels
        # --------------------------------------------------
        for i, ch in enumerate(channels):
            y = data[ch] + i * offset
            ax.plot(self.emg.time, y, label=f"Ch {ch}")

        # --------------------------------------------------
        # Plot triggers
        # --------------------------------------------------
        if show_triggers and self.emg.trigger is not None:
            if trigger_kwargs is None:
                trigger_kwargs = dict(
                    color="red",
                    linestyle="--",
                    alpha=0.4,
                    linewidth=1,
                )

            _, _, event_t = self.emg.trigger.get_events()
            for t_ev in event_t:
                ax.axvline(t_ev, **trigger_kwargs)

        # --------------------------------------------------
        # Axes cosmetics
        # --------------------------------------------------
        if tlim is not None:
            ax.set_xlim(*tlim)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG (µV)")
        ax.set_title("Continuous EMG")
        ax.legend(loc="upper right")

        return ax
