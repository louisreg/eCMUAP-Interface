from probeinterface import Probe
import numpy as np

# Exemple de positions (mm ou um selon ton choix)
positions = np.array([
    [0.0, 0.0],
    [0.0, 0.1],
    [0.0, 0.2]
])

# Création d’un probe 2D avec 3 contacts
probe = Probe(ndim=2, si_units='mm')
probe.set_contacts(positions=positions)

# Fixer l’ordre des indices de canaux (mapping vers l’acquis)
probe.set_device_channel_indices([0, 1, 2])

print(probe.get_contact_count())  # Nombre de contacts
