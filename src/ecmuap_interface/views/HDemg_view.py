from __future__ import annotations 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ecmuap_interface.core.HD_emg import HDEMG
from .eCMUAP_view  import eCMUAPView
from typing import Tuple
from ecmuap_interface.core.eCMUAP import eCMUAP




class HDEMGView:
    """
    Visualization utilities for HD-EMG data.

    Handles spatial plotting using probe geometry.
    """

    def __init__(self, hd_emg: HDEMG):
        self.hd_emg = hd_emg
        self.emg = hd_emg.emg
        self.probe = hd_emg.probe
        if self.probe.contact_positions is None:
            raise ValueError("Probe has no contact positions")

    # ==================================================
    # BASIC PROBE PLOT
    # ==================================================
    def plot_probe(self, ax=None, show_ids: bool = True):
        """
        Plot probe geometry only.
        """
        if ax is None:
            fig, ax = plt.subplots()

        pos = self.probe.contact_positions

        ax.scatter(pos[:, 0], pos[:, 1], s=60, c="white", edgecolors="k")

        if show_ids:
            for i, (x, y) in enumerate(pos):
                ax.text(x, y, str(i), ha="center", va="center")

        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("Probe geometry")

        return ax

    # ==================================================
    # METRIC HEATMAP / SCATTER
    # ==================================================
    def plot_metric(
        self,
        values: np.ndarray,
        interpolate: bool = True,
        n_interp: int = 100,
        axis_off: bool = False,
        cmap: str = "viridis",
        ax=None,
    ):
        """
        Plot a spatial metric either:
        - interpolated heatmap
        - or directly on electrode positions

        Parameters
        ----------
        values : np.ndarray
            Shape (n_channels,)
        interpolate : bool
            If True, interpolate spatially
            If False, scatter on electrodes
        n_interp : int
            Interpolation grid resolution
        """

        if values.ndim != 1:
            raise ValueError("values must be 1D (n_channels,)")

        pos = self.probe.contact_positions
        x = pos[:, 0]
        y = pos[:, 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 6))
        else:
            fig = ax.figure

        # --------------------------------------------------
        # AXIS LIMITS WITH MARGIN (KEY FIX)
        # --------------------------------------------------
        pad_x = 0.1 * (x.max() - x.min())
        pad_y = 0.1 * (y.max() - y.min())

        xlim = (x.min() - pad_x, x.max() + pad_x)
        ylim = (y.min() - pad_y, y.max() + pad_y)

        # -----------------------------
        # DIRECT ELECTRODE SCATTER
        # -----------------------------
        if not interpolate:
            sc = ax.scatter(
                x,
                y,
                c=values,
                cmap=cmap,
                s=80,
                edgecolors="k",
            )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")

            cbar = plt.colorbar(
                sc,
                ax=ax,
                fraction=0.035,
                pad=0.02,
            )

        # -----------------------------
        # INTERPOLATED HEATMAP
        # -----------------------------
        else:
            grid_x, grid_y = np.mgrid[
                x.min():x.max():complex(n_interp),
                y.min():y.max():complex(n_interp),
            ]

            # Drop NaN electrodes (e.g. positions outside the source probe
            # convex hull after spatial interpolation) so they don't corrupt
            # the Delaunay triangulation used by griddata "cubic".
            valid = np.isfinite(values)
            interp = griddata(
                pos[valid],
                values[valid],
                (grid_x, grid_y),
                method="cubic",
            )

            im = ax.imshow(
                interp.T,
                origin="lower",
                extent=(x.min(), x.max(), y.min(), y.max()),
                cmap=cmap,
            )

            ax.scatter(x, y, c="white", s=40, edgecolors="k")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")

            cbar = plt.colorbar(
                im,
                ax=ax,
                fraction=0.035,
                pad=0.02,
            )

        # --------------------------------------------------
        # AXES / LABELS
        # --------------------------------------------------
        if axis_off:
            ax.axis("off")
        else:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")

        return fig, ax


    # ==================================================
    # TIME SNAPSHOT
    # ==================================================
    def plot_snapshot(
        self,
        t_idx: int,
        raw: bool = False,
        interpolate: bool = True,
        n_interp: int = 100,
        axis_off: bool = False,
        cmap: str = "viridis",
        ax=None,
    ):
        """
        Plot instantaneous spatial snapshot of EMG.

        Parameters
        ----------
        t_idx : int
            Sample index
        """

        if t_idx < 0 or t_idx >= self.emg.n_samples:
            raise IndexError("Invalid t_idx")

        data = self.emg.raw if raw else self.emg.data
        values = data[:, t_idx]

        return self.plot_metric(
            values,
            interpolate=interpolate,
            n_interp=n_interp,
            axis_off=axis_off,
            cmap=cmap,
            ax=ax,
        )

    # ==================================================
    # eCMUAP METRIC HEATMAP
    # ==================================================
    def plot_eCMUAP_metric(
        self,
        metric: str,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        interpolate: bool = True,
        n_interp: int = 100,
        cmap: str = "viridis",
        axis_off: bool = False,
        ax=None,
        reject: bool = False,
        reject_method: str = "amplitude",
        reject_threshold: float | None = None,
        reject_auto_scale: float = 3.0,
    ):
        """
        Plot a spatial heatmap of an eCMUAP metric.

        Parameters
        ----------
        metric : str
            eCMUAP metric name: ``'rms'``, ``'peak2peak'``, ``'latency'``,
            ``'duration'``, ``'max'``, ``'min'``, …
        t_pre, t_post : float
            Epoch window around the trigger (s).
        skip_start, skip_end : int
            Trigger events to discard at the edges.
        reject : bool
            Apply automatic epoch rejection before averaging.
        reject_method, reject_threshold, reject_auto_scale
            Passed to :meth:`~ecmuap_interface.views.eCMUAP_view.eCMUAPView.reject_epochs`.
        interpolate : bool
            True → smooth heatmap;  False → electrode scatter.
        """

        # --- extract averaged eCMUAPs
        cmuap_view = eCMUAPView(self.emg)
        avg = cmuap_view.average(
            t_pre=t_pre,
            t_post=t_post,
            skip_start=skip_start,
            skip_end=skip_end,
            reject=reject,
            reject_method=reject_method,
            reject_threshold=reject_threshold,
            reject_auto_scale=reject_auto_scale,
        )

        # --- time vector
        n_samples = avg.shape[1]
        epoch_time = np.linspace(
            -t_pre,
            t_post,
            n_samples,
            endpoint=False,
        )

        # --- compute metric per channel
        # Validate metric name on a dummy object before looping
        _dummy = eCMUAP(avg[0], epoch_time)
        if not hasattr(type(_dummy), metric) and not hasattr(_dummy, metric):
            raise ValueError(
                f"Invalid eCMUAP metric '{metric}'. "
                f"Available: rms, peak2peak, peak, latency, duration, ttmin, ttmax, ..."
            )

        values = np.full(self.emg.n_channels, np.nan)

        for ch in range(self.emg.n_channels):
            ec = eCMUAP(avg[ch], epoch_time)
            try:
                values[ch] = getattr(ec, metric)
            except (IndexError, ZeroDivisionError, ValueError):
                values[ch] = np.nan

        # --- delegate plotting
        return self.plot_metric(
            values=values,
            interpolate=interpolate,
            n_interp=n_interp,
            cmap=cmap,
            axis_off=axis_off,
            ax=ax,
        )
    
    # ==================================================
    # INTERNAL: create probe-ordered figure
    # ==================================================
    def _create_probe_axes(
        self,
        label: bool = True,
        axis_off: bool = False,
        figsize: Tuple[int, int] = (10, 12),
    ):
        """
        Create a figure with one Axes per electrode,
        spatially arranged according to probe geometry.

        Parameters
        ----------
        label : bool
            Display channel labels
        axis_off : bool
            Hide all axes (ticks, spines, frames)
        figsize : tuple
            Figure size
        """

        pos = self.probe.contact_positions  # (n_channels, 2)
        x = pos[:, 0]
        y = pos[:, 1]

        # normalize to grid
        x_unique = np.unique(x)
        y_unique = np.unique(y)

        x_map = {v: i for i, v in enumerate(sorted(x_unique))}
        y_map = {v: i for i, v in enumerate(sorted(y_unique, reverse=True))}

        n_col = len(x_unique)
        n_row = len(y_unique)

        fig, axs = plt.subplots(
            n_row,
            n_col,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        # ensure 2D array
        axs = np.atleast_2d(axs)

        # turn off everything by default
        for ax in axs.ravel():
            ax.axis("off")

        out_axes = [None] * self.emg.n_channels

        for ch in range(self.emg.n_channels):
            ix = x_map[x[ch]]
            iy = y_map[y[ch]]

            ax = axs[iy, ix]
            out_axes[ch] = ax

            if not axis_off:
                ax.axis("on")

                if label:
                    ax.set_title(f"Ch {ch}", fontsize=8)
            else:
                # enforce full hiding
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return fig, out_axes

    # ==================================================
    # RAW EMG
    # ==================================================
    def plot_raw(self, label: bool = True, figsize: Tuple[int, int] = (10, 12), axis_off: bool = True, **plot_kwargs):
        fig, axs = self._create_probe_axes(label,axis_off,figsize)

        ymin, ymax = [], []

        for ch, ax in enumerate(axs):
            y = self.emg.raw[ch]
            if np.all(np.isnan(y)):
                continue
            ax.plot(self.emg.time, y, **plot_kwargs)
            ymin.append(np.nanmin(y))
            ymax.append(np.nanmax(y))

        if ymin and ymax:
            ylim_min, ylim_max = np.nanmin(ymin), np.nanmax(ymax)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                for ax in axs:
                    ax.set_ylim(1.1 * ylim_min, 1.1 * ylim_max)

        fig.suptitle("HD-EMG raw signals")
        return fig, axs

    # ==================================================
    # FILTERED EMG
    # ==================================================
    def plot_data(self, label: bool = True, figsize: Tuple[int, int] = (10, 12), axis_off: bool = True, **plot_kwargs):
        fig, axs = self._create_probe_axes(label,axis_off,figsize)

        ymin, ymax = [], []

        for ch, ax in enumerate(axs):
            y = self.emg.data[ch]
            if np.all(np.isnan(y)):
                continue
            ax.plot(self.emg.time, y, **plot_kwargs)
            ymin.append(np.nanmin(y))
            ymax.append(np.nanmax(y))

        if ymin and ymax:
            ylim_min, ylim_max = np.nanmin(ymin), np.nanmax(ymax)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                for ax in axs:
                    ax.set_ylim(1.1 * ylim_min, 1.1 * ylim_max)

        fig.suptitle("HD-EMG filtered signals")
        return fig, axs

    # ==================================================
    # INDIVIDUAL eCMUAPs
    # ==================================================
    def plot_eCMUAPs(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        label: bool = True,
        figsize: Tuple[int, int] = (10, 12),
        axis_off: bool = True,
        **plot_kwargs,
    ):
        cmuap = eCMUAPView(self.emg)
        epochs = cmuap.epochs(t_pre, t_post, skip_start, skip_end)

        epoch_time = np.linspace(
            -t_pre,
            t_post,
            epochs.shape[2],
            endpoint=False,
        )

        fig, axs = self._create_probe_axes(label,axis_off,figsize)

        ymin, ymax = [], []

        for ch, ax in enumerate(axs):
            for ev in range(epochs.shape[0]):
                y = epochs[ev, ch]
                if np.all(np.isnan(y)):
                    continue
                ax.plot(epoch_time, y, **plot_kwargs)
                ymin.append(np.nanmin(y))
                ymax.append(np.nanmax(y))

        for ax in axs:
            ax.set_xlim(epoch_time[0], epoch_time[-1])
        if ymin and ymax:
            ylim_min, ylim_max = np.nanmin(ymin), np.nanmax(ymax)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                for ax in axs:
                    ax.set_ylim(1.1 * ylim_min, 1.1 * ylim_max)

        fig.suptitle("HD-eCMUAPs (individual trials)")
        return fig, axs

    # ==================================================
    # AVERAGE eCMUAP
    # ==================================================
    def plot_avg_eCMUAP(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        label: bool = True,
        figsize: Tuple[int, int] = (10, 12),
        axis_off: bool = True,
        reject: bool = False,
        reject_method: str = "amplitude",
        reject_threshold: float | None = None,
        reject_auto_scale: float = 3.0,
        **plot_kwargs,
    ):
        cmuap = eCMUAPView(self.emg)
        avg = cmuap.average(
            t_pre, t_post, skip_start, skip_end,
            reject=reject,
            reject_method=reject_method,
            reject_threshold=reject_threshold,
            reject_auto_scale=reject_auto_scale,
        )

        epoch_time = np.linspace(
            -t_pre,
            t_post,
            avg.shape[1],
            endpoint=False,
        )

        fig, axs = self._create_probe_axes(label,axis_off,figsize)

        ymin, ymax = [], []

        for ch, ax in enumerate(axs):
            y = avg[ch]
            if np.all(np.isnan(y)):
                continue
            ax.plot(epoch_time, y, **plot_kwargs)
            ymin.append(np.nanmin(y))
            ymax.append(np.nanmax(y))

        for ax in axs:
            ax.set_xlim(epoch_time[0], epoch_time[-1])
        if ymin and ymax:
            ylim_min, ylim_max = np.nanmin(ymin), np.nanmax(ymax)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                for ax in axs:
                    ax.set_ylim(1.1 * ylim_min, 1.1 * ylim_max)

        fig.suptitle("HD-eCMUAP average")
        return fig, axs
    

    def compute_unit_to_fig_scale(self,ax):
        """
        Compute conversion factors from data units to figure fraction
        based on an Axes geometry.

        Returns
        -------
        x_unit_to_fig : float
            data-x unit → figure fraction
        y_unit_to_fig : float
            data-y unit → figure fraction
        """

        # Axes position in figure coordinates
        bbox = ax.get_position()
        fig_w = bbox.width
        fig_h = bbox.height

        # Data limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        data_w = abs(xlim[1] - xlim[0])
        data_h = abs(ylim[1] - ylim[0])

        if data_w == 0 or data_h == 0:
            raise ValueError("Axis limits are degenerate")

        x_unit_to_fig = fig_w / data_w
        y_unit_to_fig = fig_h / data_h

        return x_unit_to_fig, y_unit_to_fig
        
    def add_scale_bar(
        self,
        fig: plt.Figure,
        x_size: float,
        y_size: float,
        x_label: str,
        y_label: str,
        loc: str = "upper left",
        pad: float = 0.03,
        linewidth: float = 2.0,
        fontsize: int = 9,
    ):
        """
        Add a physical x/y scale bar to a figure (figure coordinates).
        Y axis is oriented upward.
        """

        # Convert physical sizes → figure fraction
        x_unit_to_fig, y_unit_to_fig = self.compute_unit_to_fig_scale(fig.axes[0])
        bar_x = x_size * x_unit_to_fig
        bar_y = y_size * y_unit_to_fig

        if loc == "upper left":
            x0, y0 = pad, 1 - pad
            ha_x = "left"
            ha_y = "right"

        elif loc == "upper right":
            x0, y0 = 1 - pad, 1 - pad
            bar_x *= -1
            ha_x = "right"
            ha_y = "left"

        else:
            raise ValueError("loc must be 'upper left' or 'upper right'")

        # Invisible overlay axes
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # --- X scale bar (horizontal)
        ax.plot(
            [x0, x0 + bar_x],
            [y0, y0],
            lw=linewidth,
            color="k",
            transform=fig.transFigure,
            clip_on=False,
        )
        ax.text(
            x0 + bar_x / 2,
            y0 - 0.015,
            x_label,
            ha="center",
            va="top",
            fontsize=fontsize,
            transform=fig.transFigure,
        )

        # --- Y scale bar (UPWARD)
        ax.plot(
            [x0, x0],
            [y0, y0 + bar_y],
            lw=linewidth,
            color="k",
            transform=fig.transFigure,
            clip_on=False,
        )
        ax.text(
            x0 - 0.01 if loc == "upper left" else x0 + 0.01,
            y0 + bar_y / 2,
            y_label,
            ha=ha_y,
            va="center",
            rotation=90,
            fontsize=fontsize,
            transform=fig.transFigure,
        )

    # ==================================================
    # eCMUAP AMPLITUDE MAPS
    # ==================================================
    def plot_rms_map(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        reject: bool = False,
        interpolate: bool = True,
        n_interp: int = 100,
        cmap: str = "hot",
        ax=None,
    ):
        """
        Spatial heatmap of RMS amplitude of the average eCMUAP.

        Parameters
        ----------
        t_pre, t_post : float
            Epoch window around the trigger (s).
        reject : bool
            Apply automatic epoch rejection before averaging.
        interpolate : bool
            True → smooth heatmap;  False → scatter on electrodes.
        cmap : str
            Colormap. Default ``"hot"``.

        Returns
        -------
        ax : plt.Axes
        """
        return self.plot_eCMUAP_metric(
            "rms", t_pre, t_post,
            skip_start=skip_start, skip_end=skip_end,
            interpolate=interpolate, n_interp=n_interp,
            cmap=cmap, ax=ax,
            reject=reject,
        )

    def plot_ptp_map(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        reject: bool = False,
        interpolate: bool = True,
        n_interp: int = 100,
        cmap: str = "hot",
        ax=None,
    ):
        """
        Spatial heatmap of peak-to-peak amplitude of the average eCMUAP.

        Returns
        -------
        ax : plt.Axes
        """
        return self.plot_eCMUAP_metric(
            "peak2peak", t_pre, t_post,
            skip_start=skip_start, skip_end=skip_end,
            interpolate=interpolate, n_interp=n_interp,
            cmap=cmap, ax=ax,
            reject=reject,
        )

    # ==================================================
    # PROPAGATION VELOCITY MAPS
    # ==================================================
    def plot_delay_map(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        reject: bool = False,
        interpolate: bool = True,
        n_interp: int = 100,
        cmap: str = "plasma",
        show_quiver: bool = True,
        ax=None,
    ):
        """
        Spatial heatmap of the eCMUAP negative-peak delay.

        The delay map reveals the innervation zone (IZ): electrodes near the
        IZ have the shortest delay; propagation fans outward.  A quiver
        overlay (optional) shows the local propagation direction.

        Parameters
        ----------
        show_quiver : bool
            Overlay propagation-direction arrows. Default True.

        Returns
        -------
        ax : plt.Axes
        """
        from ecmuap_interface.utils.velocity import (
            compute_delay_map, compute_velocity_map,
        )
        avg     = eCMUAPView(self.emg).average(
            t_pre, t_post, skip_start, skip_end, reject=reject,
        )
        delays  = compute_delay_map(avg, self.emg.fs)   # (n_channels,)

        fig, ax = self.plot_metric(
            delays, interpolate=interpolate, n_interp=n_interp,
            cmap=cmap, ax=ax,
        )
        ax.set_title("Delay map (ms)")

        if show_quiver and self.probe.annotations.get("pitch_um"):
            pitch = float(self.probe.annotations["pitch_um"])
            try:
                from ecmuap_interface.utils.probes import (
                    get_grid_shape_from_probe, get_grid_indices_from_probe,
                )
                n_rows, n_cols = get_grid_shape_from_probe(self.probe)
                delay_grid = np.full((n_rows, n_cols), np.nan)
                grid_idx   = get_grid_indices_from_probe(self.probe)
                for ch, (r, c) in grid_idx.items():
                    delay_grid[r, c] = delays[ch]

                speed, vx, vy = compute_velocity_map(delay_grid, pitch)

                pos = self.probe.contact_positions
                x_u = np.unique(np.round(pos[:, 0], 4))
                y_u = np.unique(np.round(pos[:, 1], 4))
                X, Y = np.meshgrid(x_u, y_u)
                mag  = np.sqrt(vx**2 + vy**2 + 1e-12)
                fm   = np.isfinite(vx) & np.isfinite(vy)
                ax.quiver(
                    X[fm], Y[fm],
                    (vx / mag)[fm], (vy / mag)[fm],
                    scale=25, width=0.005, color="white", alpha=0.75,
                )
            except Exception:
                pass   # probe is not a uniform grid — skip quiver silently

        return fig, ax

    def plot_cv_map(
        self,
        t_pre: float,
        t_post: float,
        skip_start: int = 0,
        skip_end: int = 0,
        reject: bool = False,
        axis: int = 0,
        cmap: str = "RdBu_r",
        mark_iz: bool = True,
        ax=None,
    ):
        """
        Spatial heatmap of cross-correlation conduction velocity (m/s).

        Requires the probe to be a **uniform grid** (call
        :func:`~ecmuap_interface.utils.probes.make_uniform_probe_from_base`
        before building the HDEMG object).

        Parameters
        ----------
        axis : {0, 1}
            0 → CV along rows (fibre direction = ↓);
            1 → CV along columns.
        mark_iz : bool
            Mark sign-flip positions (innervation zone candidates) with
            red crosses. Default True.

        Returns
        -------
        fig : plt.Figure
        ax  : plt.Axes
        """
        from ecmuap_interface.utils.velocity import compute_xcorr_cv
        from ecmuap_interface.utils.probes import reshape_to_grid, is_uniform_grid

        if not is_uniform_grid(self.probe):
            raise ValueError(
                "plot_cv_map requires a uniform-grid probe. "
                "Interpolate first with HDEMG.interpolate_to_probe()."
            )

        pitch = float(self.probe.annotations.get("pitch_um", 0))
        if pitch == 0:
            raise ValueError("Probe has no 'pitch_um' annotation.")

        avg      = eCMUAPView(self.emg).average(
            t_pre, t_post, skip_start, skip_end, reject=reject,
        )
        avg_grid = reshape_to_grid(avg, self.probe)
        cv_map, _ = compute_xcorr_cv(avg_grid, pitch, self.emg.fs, axis)

        pos = self.probe.contact_positions
        x_u = np.unique(np.round(pos[:, 0], 4))
        y_u = np.unique(np.round(pos[:, 1], 4))

        if axis == 0:
            y_mid   = (y_u[:-1] + y_u[1:]) / 2
            extent  = (x_u[0] - pitch/2, x_u[-1] + pitch/2,
                       y_mid[-1] + pitch/2, y_mid[0] - pitch/2)
        else:
            x_mid   = (x_u[:-1] + x_u[1:]) / 2
            extent  = (x_mid[0] - pitch/2, x_mid[-1] + pitch/2,
                       y_u[-1] + pitch/2, y_u[0] - pitch/2)

        absmax = np.nanpercentile(np.abs(cv_map[np.isfinite(cv_map)]), 95) * 1.2 \
                 if np.any(np.isfinite(cv_map)) else 10.0

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 5))
        else:
            fig = ax.figure

        im = ax.imshow(
            cv_map, origin="upper", extent=extent,
            cmap=cmap, vmin=-absmax, vmax=absmax, aspect="auto",
        )
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="CV (m/s)")

        if mark_iz:
            if axis == 0:
                for ci, cx in enumerate(x_u):
                    col = cv_map[:, ci]
                    valid = np.where(np.isfinite(col))[0]
                    if len(valid) > 1:
                        flips = valid[np.where(np.diff(np.sign(col[valid])) != 0)[0]]
                        for f in flips:
                            ax.plot(cx, y_mid[f], "r+",
                                    markersize=10, markeredgewidth=1.5,
                                    label="IZ candidate")
            else:
                for ri, ry in enumerate(y_u):
                    row = cv_map[ri, :]
                    valid = np.where(np.isfinite(row))[0]
                    if len(valid) > 1:
                        flips = valid[np.where(np.diff(np.sign(row[valid])) != 0)[0]]
                        for f in flips:
                            ax.plot(x_mid[f], ry, "r+",
                                    markersize=10, markeredgewidth=1.5)

        ax.scatter(pos[:, 0], pos[:, 1], c="white", s=10,
                   edgecolors="gray", linewidths=0.4, zorder=5)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(f"CV map — axis {axis} (m/s)")
        return fig, ax

    # ==================================================
    # SPATIAL ANIMATION
    # ==================================================
    def animate_snapshot(
        self,
        t_start: float,
        t_end: float,
        fps: int = 25,
        stride: int | None = None,
        n_interp: int = 100,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        raw: bool = False,
        ax: plt.Axes | None = None,
        show_time: bool = True,
    ):
        """
        Animate spatial heatmap snapshots over a time window.

        All frames are precomputed in a single griddata call for performance.
        The colormap is diverging and zero-centred by default so that positive
        and negative deflections are directly readable.

        Parameters
        ----------
        t_start : float
            Start of the animation window (s, absolute signal time).
        t_end : float
            End of the animation window (s).
        fps : int
            Animation playback frame rate. Default: 25.
        stride : int or None
            Show every *stride*-th sample.  None → auto (~150 frames total).
        n_interp : int
            Spatial interpolation grid resolution. Default: 100.
        cmap : str
            Matplotlib colormap. Default: ``"RdBu_r"`` (diverging).
        vmin, vmax : float or None
            Colour scale limits. None → symmetric ±|max| across the window.
        raw : bool
            If True, use raw signal. Default: False.
        ax : plt.Axes or None
            Existing axes to draw into. None → new figure.
        show_time : bool
            Overlay current time (ms) as a text annotation. Default: True.

        Returns
        -------
        fig : plt.Figure
        anim : matplotlib.animation.FuncAnimation
        """
        from matplotlib.animation import FuncAnimation

        if stride is None:
            n_raw = max(1, int(round((t_end - t_start) * self.emg.fs)))
            stride = max(1, n_raw // 150)

        d = _precompute_interp_frames(self, t_start, t_end, stride, n_interp, raw)

        finite = d["frames"][np.isfinite(d["frames"])]
        if vmin is None and vmax is None and len(finite) > 0:
            abs_max = np.abs(finite).max()
            vmin, vmax = -abs_max, abs_max

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 5))
        else:
            fig = ax.figure

        x, y = d["x"], d["y"]
        im = ax.imshow(
            d["frames"][:, :, 0].T,
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
        )
        ax.scatter(x, y, c="white", s=15, edgecolors="k", linewidths=0.5, zorder=5)

        pad_x = max(0.1 * (x.max() - x.min()), 500.0)
        pad_y = max(0.1 * (y.max() - y.min()), 500.0)
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
        ax.axis("off")

        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="µV")

        time_text = None
        if show_time:
            t0_ms = d["frame_indices"][0] / d["fs"] * 1e3
            time_text = ax.text(
                0.05, 0.95, f"t = {t0_ms:.2f} ms",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        n_frames = d["frames"].shape[2]

        def _update(i: int):
            im.set_data(d["frames"][:, :, i].T)
            artists = [im]
            if time_text is not None:
                t_ms = d["frame_indices"][i] / d["fs"] * 1e3
                time_text.set_text(f"t = {t_ms:.2f} ms")
                artists.append(time_text)
            return artists

        anim = FuncAnimation(
            fig, _update, frames=n_frames, interval=1000.0 / fps, blit=True,
        )
        return fig, anim


# ==================================================
# MODULE-LEVEL HELPERS (animation)
# ==================================================

def _precompute_interp_frames(
    view: HDEMGView,
    t_start: float,
    t_end: float,
    stride: int,
    n_interp: int,
    raw: bool = False,
) -> dict:
    """
    Precompute spatially interpolated frames for animation.

    Uses a single griddata call on all time frames simultaneously
    (griddata supports (n_pts, k) values → output (n_interp, n_interp, k)).

    Returns a dict with keys:
        frames          (n_interp, n_interp, n_frames)
        frame_indices   (n_frames,) — original sample indices
        fs              sampling frequency (Hz)
        x, y            electrode positions (µm)
        grid_x, grid_y  interpolation meshgrid
    """
    data = view.emg.raw if raw else view.emg.data
    fs   = view.emg.fs

    i_start = max(0, int(round(t_start * fs)))
    i_end   = min(view.emg.n_samples, int(round(t_end * fs)))
    frame_indices = np.arange(i_start, i_end, stride)

    pos = view.probe.contact_positions
    x, y = pos[:, 0], pos[:, 1]

    grid_x, grid_y = np.mgrid[
        x.min():x.max():complex(n_interp),
        y.min():y.max():complex(n_interp),
    ]

    window = data[:, frame_indices]                   # (n_ch, n_frames)
    valid  = np.all(np.isfinite(window), axis=1)      # channels finite everywhere

    if valid.sum() < 4:
        raise ValueError(
            f"Only {valid.sum()} finite channels in the requested window "
            f"[{t_start:.4f}s – {t_end:.4f}s] — cannot interpolate."
        )

    # Single call: values (n_valid, n_frames) → frames (n_interp, n_interp, n_frames)
    frames = griddata(
        pos[valid], window[valid], (grid_x, grid_y), method="cubic",
    )

    return dict(
        frames=frames,
        frame_indices=frame_indices,
        fs=fs,
        x=x, y=y,
        grid_x=grid_x, grid_y=grid_y,
    )


def animate_snapshot_comparison(
    views: list[tuple[str, "HDEMGView"]],
    t_start: float,
    t_end: float,
    fps: int = 25,
    stride: int | None = None,
    n_interp: int = 100,
    cmap: str = "RdBu_r",
    n_cols: int = 3,
    figsize: tuple | None = None,
    show_time: bool = True,
    shared_scale: bool = True,
    surface: bool = False,
    elev: float = 35.0,
    azim: float = -50.0,
):
    """
    Animate spatial heatmaps for multiple views side by side.

    Parameters
    ----------
    views : list of (label: str, view: HDEMGView)
        One entry per panel.
    t_start, t_end : float
        Animation time window (s, absolute signal time).
    fps : int
        Playback frame rate. Default: 25.
    stride : int or None
        Use every *stride*-th sample. None → auto (~150 frames per view).
    n_interp : int
        Spatial grid resolution. Default: 100.
    cmap : str
        Colormap. Default: ``"RdBu_r"`` (diverging, zero-centred).
    n_cols : int
        Number of columns in the panel grid. Default: 3.
    figsize : tuple or None
        Figure size. None → auto.
    show_time : bool
        Overlay time annotation on the first panel. Default: True.
    shared_scale : bool
        True  → one ±|max| colour scale across all views (direct comparison).
        False → each view gets its own ±|max| scale (maximises contrast).
        Default: True.
    surface : bool
        True  → 3-D surface plot (height + colour = amplitude).
        False → 2-D heatmap (imshow). Default: False.
    elev, azim : float
        3-D view elevation and azimuth angles (degrees).
        Only used when ``surface=True``.

    Returns
    -------
    fig : plt.Figure
    anim : matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mpl_colors

    n_views = len(views)
    n_rows  = (n_views + n_cols - 1) // n_cols

    # ---- Precompute all views ----
    all_data: list[dict] = []
    for _label, view in views:
        if stride is None:
            n_raw   = max(1, int(round((t_end - t_start) * view.emg.fs)))
            _stride = max(1, n_raw // 150)
        else:
            _stride = stride
        all_data.append(_precompute_interp_frames(view, t_start, t_end, _stride, n_interp))

    # ---- Colour scale (shared or per-view) ----
    if shared_scale:
        abs_max = 0.0
        for d in all_data:
            finite = d["frames"][np.isfinite(d["frames"])]
            if len(finite) > 0:
                abs_max = max(abs_max, np.abs(finite).max())
        vmins = [-abs_max] * n_views
        vmaxs = [ abs_max] * n_views
    else:
        vmins, vmaxs = [], []
        for d in all_data:
            finite = d["frames"][np.isfinite(d["frames"])]
            m = np.abs(finite).max() if len(finite) > 0 else 1.0
            vmins.append(-m)
            vmaxs.append( m)

    cmap_obj = mpl_cm.get_cmap(cmap)
    norms    = [mpl_colors.Normalize(vmin=v0, vmax=v1)
                for v0, v1 in zip(vmins, vmaxs)]

    # ==========================================================
    # 3-D SURFACE BRANCH
    # ==========================================================
    if surface:
        if figsize is None:
            figsize = (5.5 * n_cols, 5.0 * n_rows)

        fig = plt.figure(figsize=figsize)
        axs_flat: list = [
            fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            for idx in range(n_views)
        ]

        # Draw frame 0 and add a static colorbar per panel
        for i, (label, _view) in enumerate(views):
            ax   = axs_flat[i]
            d    = all_data[i]
            norm = norms[i]

            frame0   = d["frames"][:, :, 0]
            frame0_f = np.where(np.isfinite(frame0), frame0, 0.0)

            ax.plot_surface(
                d["grid_x"], d["grid_y"], frame0_f,
                facecolors=cmap_obj(norm(frame0_f)),
                shade=False, rstride=1, cstride=1, antialiased=False,
            )
            ax.set_zlim(vmins[i], vmaxs[i])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("x (µm)", fontsize=7, labelpad=0)
            ax.set_ylabel("y (µm)", fontsize=7, labelpad=0)
            ax.set_zlabel("µV",     fontsize=7, labelpad=0)
            ax.tick_params(labelsize=6)

            sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.12, label="µV")

        time_texts_3d: list = []
        if show_time:
            d0    = all_data[0]
            t0_ms = d0["frame_indices"][0] / d0["fs"] * 1e3
            tt = axs_flat[0].text2D(
                0.05, 0.95, f"t = {t0_ms:.2f} ms",
                transform=axs_flat[0].transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
            time_texts_3d.append(tt)

        n_frames = min(d["frames"].shape[2] for d in all_data)

        def _update_3d(frame_i: int):
            for j, (label, _view) in enumerate(views):
                ax   = axs_flat[j]
                d    = all_data[j]
                norm = norms[j]
                ax.clear()

                fi      = min(frame_i, d["frames"].shape[2] - 1)
                frame   = d["frames"][:, :, fi]
                frame_f = np.where(np.isfinite(frame), frame, 0.0)

                ax.plot_surface(
                    d["grid_x"], d["grid_y"], frame_f,
                    facecolors=cmap_obj(norm(frame_f)),
                    shade=False, rstride=1, cstride=1, antialiased=False,
                )
                ax.set_zlim(vmins[j], vmaxs[j])
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(label, fontsize=10)
                ax.set_xlabel("x (µm)", fontsize=7, labelpad=0)
                ax.set_ylabel("y (µm)", fontsize=7, labelpad=0)
                ax.set_zlabel("µV",     fontsize=7, labelpad=0)
                ax.tick_params(labelsize=6)

            if time_texts_3d:
                d0 = all_data[0]
                fi = min(frame_i, len(d0["frame_indices"]) - 1)
                t_ms = d0["frame_indices"][fi] / d0["fs"] * 1e3
                time_texts_3d[0].set_text(f"t = {t_ms:.2f} ms")

            return []

        anim = FuncAnimation(
            fig, _update_3d, frames=n_frames,
            interval=1000.0 / fps, blit=False,
        )
        return fig, anim

    # ==========================================================
    # 2-D HEATMAP BRANCH (imshow)
    # ==========================================================
    if figsize is None:
        figsize = (4.5 * n_cols, 5.0 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axs_flat = np.array(axs).ravel()
    for ax in axs_flat[n_views:]:
        ax.set_visible(False)

    ims: list        = []
    time_texts: list = []

    for i, (label, _view) in enumerate(views):
        ax = axs_flat[i]
        d  = all_data[i]
        x, y = d["x"], d["y"]

        im = ax.imshow(
            d["frames"][:, :, 0].T,
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap=cmap, vmin=vmins[i], vmax=vmaxs[i], aspect="auto",
        )
        ims.append(im)

        ax.scatter(x, y, c="white", s=12, edgecolors="k", linewidths=0.5, zorder=5)
        pad_x = max(0.1 * (x.max() - x.min()), 500.0)
        pad_y = max(0.1 * (y.max() - y.min()), 500.0)
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
        ax.axis("off")
        ax.set_title(label, fontsize=10)

        if show_time and i == 0:
            t0_ms = d["frame_indices"][0] / d["fs"] * 1e3
            tt = ax.text(
                0.05, 0.95, f"t = {t0_ms:.2f} ms",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
            time_texts.append(tt)

    # Colorbar(s)
    if shared_scale:
        fig.colorbar(
            ims[0], ax=axs_flat[:n_views].tolist(),
            location="right", fraction=0.02, pad=0.02, label="µV",
        )
    else:
        for i, im in enumerate(ims):
            plt.colorbar(im, ax=axs_flat[i], fraction=0.04, pad=0.02, label="µV")

    n_frames = min(d["frames"].shape[2] for d in all_data)

    def _update_2d(frame_i: int):
        artists = []
        for d, im in zip(all_data, ims):
            fi = min(frame_i, d["frames"].shape[2] - 1)
            im.set_data(d["frames"][:, :, fi].T)
            artists.append(im)
        if time_texts:
            d0 = all_data[0]
            fi = min(frame_i, len(d0["frame_indices"]) - 1)
            t_ms = d0["frame_indices"][fi] / d0["fs"] * 1e3
            time_texts[0].set_text(f"t = {t_ms:.2f} ms")
            artists.append(time_texts[0])
        return artists

    anim = FuncAnimation(
        fig, _update_2d, frames=n_frames,
        interval=1000.0 / fps, blit=True,
    )
    return fig, anim
