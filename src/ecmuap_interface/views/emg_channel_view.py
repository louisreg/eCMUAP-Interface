import matplotlib.pyplot as plt
from ..core.emg_data import EMGData


class EMGChannelView:
    """
    Visualization utilities for a single EMG channel.
    """

    def __init__(self, emg: EMGData, channel: int):
        if channel < 0 or channel >= emg.n_channels:
            raise ValueError("Invalid channel index")

        self.emg = emg
        self.channel = channel

    def plot(
        self,
        raw: bool = False,
        tlim: tuple[float, float] | None = None,
        show_triggers: bool = True,
        ax: plt.Axes | None = None,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Plot continuous EMG for one channel.

        Parameters
        ----------
        raw : bool
            Plot raw instead of filtered data
        tlim : (tmin, tmax) | None
            Time window (s)
        show_triggers : bool
            Overlay trigger events
        ax : matplotlib Axes | None
            Existing axes
        plot_kwargs :
            Passed to matplotlib plot()
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))

        data = self.emg.raw if raw else self.emg.data
        y = data[self.channel]

        ax.plot(self.emg.time, y, **plot_kwargs)

        if tlim is not None:
            ax.set_xlim(*tlim)

        if show_triggers and self.emg.trigger is not None:
            _, _, event_t = self.emg.trigger.get_events()
            for t_ev in event_t:
                ax.axvline(
                    t_ev,
                    color="red",
                    linestyle="--",
                    alpha=0.4,
                    linewidth=1,
                )

        ax.set_title(f"EMG Channel {self.channel}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG (µV)")

        return ax
