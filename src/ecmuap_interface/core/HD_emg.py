from ecmuap_interface.core.emg_data import EMGData
from ecmuap_interface.views.eCMUAP_view import eCMUAPView
from ecmuap_interface.utils.probes import is_uniform_grid, reshape_to_grid, grid_to_vector
from probeinterface import Probe
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import convolve

class HDEMG:
    def __init__(self, emg: EMGData, probe: Probe):
        if emg.n_channels != probe.get_contact_count():
            raise ValueError("EMG channels and probe contacts mismatch")

        self.emg = emg
        self.probe = probe
        self.cmuap = eCMUAPView(emg) if emg.trigger is not None else None

    @property
    def positions(self):
        return self.probe.contact_positions  # (n_channels, 2)

    @property
    def channel_map(self):
        return self.probe.device_channel_indices
    

    def interpolate_to_probe(
        self,
        target_probe,
        method: str = "cubic",
        fill_value: float = np.nan,
        raw: bool = False,
    ) -> "EMGData":
        """
        Spatially interpolate HD-EMG data to a new probe geometry.

        Parameters
        ----------
        target_probe : probeinterface.Probe
            Probe defining the target electrode positions
        method : str
            Interpolation method ("linear", "nearest", "cubic")
        fill_value : float
            Value used outside convex hull
        raw : bool
            If True, interpolate raw signal instead of processed data

        Returns
        -------
        HDEMG
            New HDEMG object with interpolated EMGData and target_probe
        """

        # -------------------------------------------------
        # Source data
        # -------------------------------------------------
        src_pos = self.probe.contact_positions      # (n_src, 2)
        tgt_pos = target_probe.contact_positions    # (n_tgt, 2)

        data_src = self.emg.raw if raw else self.emg.data
        # data_src shape: (n_src, n_samples)

        n_src, n_samples = data_src.shape
        n_tgt = tgt_pos.shape[0]

        if src_pos.shape[0] != n_src:
            raise ValueError(
                "Number of source channels does not match probe contacts"
            )

        # -------------------------------------------------
        # Allocate output
        # -------------------------------------------------
        data_interp = np.empty((n_tgt, n_samples), dtype=data_src.dtype)

        # -------------------------------------------------
        # Vectorized spatial interpolation (FAST)
        # -------------------------------------------------
        data_interp = griddata(
            src_pos,           # (n_src, 2)
            data_src,          # (n_src, n_samples)
            tgt_pos,           # (n_tgt, 2)
            method=method,
            fill_value=fill_value,
        )

        # -------------------------------------------------
        # Build new EMGData
        # -------------------------------------------------
        emg_interp = EMGData(
            data=data_interp,
            time=self.emg.time.copy(),
            trigger=self.emg.trigger,  # same trigger object
        )

        # -------------------------------------------------
        # Return new HDEMG
        # -------------------------------------------------
        return emg_interp
    

    def spatial_filter(self, kernel, raw=False):
        probe = self.probe

        if not is_uniform_grid(probe):
            raise ValueError("Spatial filtering requires a uniform grid probe")

        data_src = self.emg.raw if raw else self.emg.data

        # (ch, t) → (row, col, t)
        grid = reshape_to_grid(data_src, probe)

        # convolution 2D spatiale
        kernel_3d = kernel()[:, :, None]
        grid_filt = convolve(grid, kernel_3d, mode="nearest")

        # back to (ch, t)
        data_filt = grid_to_vector(grid_filt, probe)

        emg_filt = EMGData(
            data=data_filt,
            time=self.emg.time,
            trigger=self.emg.trigger,
            spatial_filter=kernel
        )


        return emg_filt