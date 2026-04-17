import numpy as np
import matplotlib.pyplot as plt
from probeinterface import Probe, write_probeinterface
from probeinterface.plotting import plot_probe

#https://www.neuronexus.com/files/surface/eeg/H32-EEG-Maps(32-ch).pdf
# Note: doesn't match what's there: https://www.neuronexus.com/products/electrode-arrays/surface-grids/

# ==================================================
# PARAMETERS
# ==================================================
radius = 250.0   # µm, electrode radius
offset = 100.0   # µm, safety margin around electrodes

# ==================================================
# EXPLICIT H32 -> RIPPLE CHANNEL MAPPING
# Source: NeuroNexus H32 mapping datasheet
# ==================================================
H32_TO_RIPPLE = {
    1: 31,  2: 29,  3: 27,  4: 25,
    5: 23,  6: 21,  7: 19,  8: 17,

    9: 32, 10: 30, 11: 28, 12: 26,
    13: 24, 14: 22, 15: 20, 16: 18,

    17: 16, 18: 14, 19: 12, 20: 10,
    21: 8,  22: 6,  23: 4,  24: 2,

    25: 15, 26: 13, 27: 11, 28: 9,
    29: 7,  30: 5,  31: 3,  32: 1,
}

# Ordered array (H32 contact 1 -> 32)
device_channel_indices = np.array(
    [H32_TO_RIPPLE[cid] for cid in range(1, 33)]
) - 1  # convert to 0-based indexing

# ==================================================
# EXPLICIT ELECTRODE GEOMETRY
# contact_id (H32) -> (x, y) in millimeters
# ==================================================
contacts_mm = {
    1:  (4.464, -7.0),
    2:  (0.866, -7.0),
    3:  (2.273, -6.0),
    4:  (4.464, -5.0),
    5:  (0.866, -5.0),
    6:  (2.273, -4.0),
    7:  (4.464, -3.0),
    8:  (0.866, -3.0),
    9:  (2.273, -2.0),
    10: (4.464, -1.0),
    11: (0.866, -1.0),
    12: (2.273,  0.0),
    13: (4.464,  1.0),
    14: (0.866,  1.0),
    15: (2.273,  2.0),
    16: (0.866,  3.0),

    17: (-0.866,  3.0),
    18: (-2.273,  2.0),
    19: (-0.866,  1.0),
    20: (-4.464,  1.0),
    21: (-2.273,  0.0),
    22: (-0.866, -1.0),
    23: (-4.464, -1.0),
    24: (-2.273, -2.0),
    25: (-0.866, -3.0),
    26: (-4.464, -3.0),
    27: (-2.273, -4.0),
    28: (-0.866, -5.0),
    29: (-4.464, -5.0),
    30: (-2.273, -6.0),
    31: (-0.866, -7.0),
    32: (-4.464, -7.0),
}

# ==================================================
# BUILD POSITION ARRAYS (ORDERED BY H32 ID)
# ==================================================
contact_ids = np.arange(1, 33)
positions_mm = np.array([contacts_mm[cid] for cid in contact_ids])
positions_um = positions_mm * 1e3  # mm -> µm

# ==================================================
# CREATE PROBE
# ==================================================
probe = Probe(ndim=2, si_units="um")

probe.set_contacts(
    positions=positions_um,
    shapes="circle",
    shape_params={"radius": radius}
)

probe.set_contact_ids(contact_ids)
probe.set_device_channel_indices(device_channel_indices)
probe.annotate(name="NeuroNexus_H32")

# ==================================================
# SIZE ESTIMATION (CONTACTS + RADIUS + OFFSET)
# ==================================================
xmin_c, ymin_c = positions_um.min(axis=0)
xmax_c, ymax_c = positions_um.max(axis=0)

xmin = xmin_c - radius
xmax = xmax_c + radius
ymin = ymin_c - radius
ymax = ymax_c + radius

width_um = (xmax - xmin) + 2 * offset
height_um = (ymax - ymin) + 2 * offset

print("=== Estimated probe size ===")
print(f"Width  : {width_um:.1f} µm ({width_um/1e3:.3f} mm)")
print(f"Height : {height_um:.1f} µm ({height_um/1e3:.3f} mm)")

# ==================================================
# PROBE OUTLINE
# Top profile defined by contacts: 20 -> 17 -> 16 -> 13
# ==================================================
p13 = positions_um[13 - 1]
p16 = positions_um[16 - 1]
p17 = positions_um[17 - 1]
p20 = positions_um[20 - 1]

polygon = [
    # Bottom
    (xmin - offset, ymin - offset),
    (xmax + offset, ymin - offset),

    # Right side
    (xmax + offset, p13[1]),

    # Top profile (data-driven)
    (p13[0] + offset + radius, p13[1]+radius),
    (p16[0], p16[1] + offset+radius),
    (p17[0], p17[1] + offset+radius),
    (p20[0] - offset - radius, p20[1]+radius),

    # Left side
    (xmin - offset, p20[1]),
]

probe.set_planar_contour(polygon)

# ==================================================
# PLOTTING
# ==================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

plot_probe(
    probe,
    ax=ax[0],
    with_contact_id=True,
    with_device_index=False
)
ax[0].set_title("NeuroNexus H32 – Contact IDs")
ax[0].set_xlabel("x (µm)")
ax[0].set_ylabel("y (µm)")
ax[0].set_aspect("equal")

plot_probe(
    probe,
    ax=ax[1],
    with_contact_id=False,
    with_device_index=True
)
ax[1].set_title("NeuroNexus H32 – Ripple Channels")
ax[1].set_xlabel("x (µm)")
ax[1].set_ylabel("y (µm)")
ax[1].set_aspect("equal")

write_probeinterface("NeuroNexus_H32_probe.json",probe)
plt.tight_layout()
plt.show()
