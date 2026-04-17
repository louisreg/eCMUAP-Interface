import matplotlib.pyplot as plt
import numpy as np
from probeinterface.plotting import plot_probe
from probeinterface import Probe

from ecmuap_interface.probe_lib.get_probes import NeuroNexus_H32
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

base_probe = NeuroNexus_H32()

interp_probe = make_uniform_probe_from_base(
    base_probe,
    #pitch_um=1500.0,   # grille régulière 500 µm
)

fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

plot_probe(base_probe, ax=ax[0], with_contact_id=False)
ax[0].set_title("Original H32")

plot_probe(interp_probe, ax=ax[1], with_contact_id=False)
ax[1].set_title("Uniform probe (interpolation grid)")

for a in ax:
    a.set_aspect("equal")

plt.tight_layout()
plt.show()
