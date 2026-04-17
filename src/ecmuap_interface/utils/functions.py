import numpy as np
from scipy.interpolate import CubicSpline

def smooth_start_end(signal, n_start=10, n_end=10):
    """
    Smoothly fade the start and end of a signal toward zero, with
    different fade lengths for start and end.

    Parameters
    ----------
    signal : np.ndarray
        EMG (MUAP) signal to process
    n_start : int
        Number of samples for fade-in at the beginning
    n_end : int
        Number of samples for fade-out at the end

    Returns
    -------
    np.ndarray
        Signal with smoothed start and end
    """
    signal = np.array(signal, dtype=float)
    N = len(signal)
    if n_start + n_end >= N:
        raise ValueError("n_start + n_end is too large compared to the signal length")

    # Fade-in for start
    if n_start > 0:
        taper_start = np.hanning(2 * n_start)[:n_start]  # 0 → 1
        signal[:n_start] *= taper_start

    # Fade-out for end
    if n_end > 0:
        taper_end = np.hanning(2 * n_end)[-n_end:]  # 1 → 0
        signal[-n_end:] *= taper_end

    return signal

def smooth_from_zero_crossing_to_end(x, extra_points=5):
    """
    Smooths the signal from the last zero-crossing until the end
    using a cubic spline that ensures zero slope at the end.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    extra_points : int
        Number of additional points after the zero-crossing
        to include in the smoothing region (makes transition softer).
    
    Returns
    -------
    y : np.ndarray
        Smoothed signal.
    """
    # Find last zero-crossing
    sign_changes = np.where(np.diff(np.sign(x)) != 0)[0]
    if len(sign_changes) == 0:
        return x
    idx_zero = sign_changes[-1]  # index before zero crossing
    idx_zero += 1  # move to first point after crossing

    # Extend smoothing region
    idx_start = max(0, idx_zero - extra_points)
    t = np.arange(len(x))

    # Points for spline fitting
    fit_t = np.array([idx_start, len(x)-1])
    fit_y = np.array([x[idx_start], 0])  # fade to 0 at the end

    # Fit cubic spline with natural BC (zero slope at end)
    cs = CubicSpline(fit_t, fit_y, bc_type=('natural', (1, 0.0)))

    # Replace the end with spline values
    y = x.copy()
    y[idx_start:] = cs(t[idx_start:])
    return y


def is_clipped(x, clip_thresh=None, clip_frac_thresh=0.01):
    """
    Detect clipping/saturation.
    - clip_thresh: numeric threshold for amplitude considered as clip (if None use 99.5 percentile)
    - clip_frac_thresh: fraction of samples at/above clip threshold to declare clipping
    """
    if clip_thresh is None:
        clip_thresh = np.percentile(np.abs(x), 99.5)
    n_clip = np.sum(np.abs(x) >= clip_thresh)
    return (n_clip / len(x)) >= clip_frac_thresh, clip_thresh

