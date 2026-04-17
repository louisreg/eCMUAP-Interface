import pandas as pd
import numpy as np
from numpy.typing import NDArray
from probeinterface import Probe

def Ripple_to_array(df: pd.DataFrame, probe: Probe) -> NDArray[np.float64]:
    """Convert Ripple DataFrame to multi-channel array ordered by probe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Ripple data with columns like 'raw 1', 'raw 2', etc.
    probe : Probe
        Probe object with device_channel_indices defined.

    Returns
    -------
    NDArray[np.float_]
        Multi-channel array of shape (n_channels, n_samples) ordered by probe.
    """
    if probe.device_channel_indices is None:
        raise ValueError("Probe must have device_channel_indices defined.")

    device_channels = probe.device_channel_indices  # 0-based indexing
    data = []

    for ch in device_channels:
        col = f"raw {ch + 1}"  # Ripple uses 1-based indexing
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in DataFrame")
        data.append(df[col].values)

    return np.vstack(data)  # shape = (n_channels, n_samples)       