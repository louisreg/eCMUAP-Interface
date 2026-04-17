import numpy as np
from probeinterface import Probe
from matplotlib.path import Path

def compute_min_pitch(positions: np.ndarray) -> float:
    """
    Compute minimal inter-electrode distance.
    """
    n = positions.shape[0]
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d > 0:
                dmin = min(dmin, d)
    return dmin


def make_uniform_probe_from_base(
    base_probe: Probe,
    name_suffix: str = "_uniform",
) -> Probe:
    """
    Create a uniformly sampled probe from a base probe:
    - same contour
    - same electrode radius
    - pitch = minimal inter-electrode distance
    - grid aligned on real electrodes
    """

    # --------------------------------------------------
    # BASE DATA
    # --------------------------------------------------
    pos = base_probe.contact_positions
    contour = np.asarray(base_probe.probe_planar_contour)
    if contour is None:
        raise ValueError("Base probe has no planar contour")

    contour_path = Path(contour)

    # Electrode radius
    radius = float(base_probe.contact_shape_params[0]["radius"])

    # --------------------------------------------------
    # COMPUTE PITCH (automatic)
    # --------------------------------------------------
    pitch = compute_min_pitch(pos)

    # --------------------------------------------------
    # GRID ALIGNMENT
    # --------------------------------------------------
    # Use bottom-left electrode as anchor
    ref_idx = np.lexsort((pos[:, 0], pos[:, 1]))[0]
    x0, y0 = pos[ref_idx]

    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)

    # Align grid on electrode lattice
    kx_min = int(np.floor((xmin - x0) / pitch))
    kx_max = int(np.ceil((xmax - x0) / pitch))
    ky_min = int(np.floor((ymin - y0) / pitch))
    ky_max = int(np.ceil((ymax - y0) / pitch))

    xs = x0 + pitch * np.arange(kx_min, kx_max + 1)
    ys = y0 + pitch * np.arange(ky_min, ky_max + 1)

    X, Y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # --------------------------------------------------
    # KEEP POINTS INSIDE PROBE OUTLINE
    # --------------------------------------------------
    inside = contour_path.contains_points(grid_points)
    positions_uniform = grid_points[inside]

    # --------------------------------------------------
    # CREATE NEW PROBE
    # --------------------------------------------------
    probe = Probe(ndim=2, si_units=base_probe.si_units)

    probe.set_contacts(
        positions=positions_uniform,
        shapes="circle",
        shape_params={"radius": radius},
    )

    probe.set_contact_ids(
        np.arange(1, positions_uniform.shape[0] + 1)
    )

    probe.set_planar_contour(contour)

    base_name = base_probe.annotations.get("name", "probe")
    probe.annotate(
        name=base_name + name_suffix,
        derived_from=base_name,
        pitch_um=pitch,
    )

    return probe



def is_uniform_grid(probe, tol=1e-6):
    pos = probe.contact_positions
    x = np.unique(np.round(pos[:, 0], 6))
    y = np.unique(np.round(pos[:, 1], 6))

    # distances
    dx = np.diff(np.sort(x))
    dy = np.diff(np.sort(y))

    if len(dx) == 0 or len(dy) == 0:
        return False

    return (
        np.all(np.abs(dx - dx[0]) < tol) and
        np.all(np.abs(dy - dy[0]) < tol)
    )


def get_grid_shape_from_probe(probe):
    pos = probe.contact_positions
    x_unique = np.unique(np.round(pos[:, 0], 6))
    y_unique = np.unique(np.round(pos[:, 1], 6))

    n_col = len(x_unique)
    n_row = len(y_unique)

    return n_row, n_col


def get_grid_indices_from_probe(probe):
    """
    Returns
    -------
    grid_indices : dict
        ch -> (row, col)
    """
    pos = probe.contact_positions

    x_unique = np.unique(np.round(pos[:, 0], 6))
    y_unique = np.unique(np.round(pos[:, 1], 6))[::-1]  # y up

    x_map = {x: i for i, x in enumerate(x_unique)}
    y_map = {y: i for i, y in enumerate(y_unique)}

    grid_indices = {}

    for ch, (x, y) in enumerate(pos):
        col = x_map[np.round(x, 6)]
        row = y_map[np.round(y, 6)]
        grid_indices[ch] = (row, col)

    return grid_indices


def reshape_to_grid(data, probe):
    """
    data : (n_channels, n_samples)
    """
    n_row, n_col = get_grid_shape_from_probe(probe)
    grid_idx = get_grid_indices_from_probe(probe)

    n_samples = data.shape[1]
    grid = np.full((n_row, n_col, n_samples), np.nan)

    for ch, (r, c) in grid_idx.items():
        grid[r, c, :] = data[ch, :]

    return grid

def grid_to_vector(grid, probe):
    n_ch = probe.contact_positions.shape[0]
    n_samples = grid.shape[2]

    data = np.zeros((n_ch, n_samples))
    grid_idx = get_grid_indices_from_probe(probe)

    for ch, (r, c) in grid_idx.items():
        data[ch, :] = grid[r, c, :]

    return data
