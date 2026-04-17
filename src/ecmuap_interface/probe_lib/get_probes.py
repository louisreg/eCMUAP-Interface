from probeinterface import Probe, read_probeinterface
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent

def NeuroNexus_H32() -> Probe:
    probe_file = BASE_DIR / "probes" / "NeuroNexus_H32_probe.json"
    probes = read_probeinterface(probe_file)
    return probes.probes[0]