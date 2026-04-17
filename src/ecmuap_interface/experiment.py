class Experiment:
    #Not ready, placeholder for future development
    def __init__(self, name, probe, metadata=None):
        self.name = name
        self.probe = probe
        self.metadata = metadata or {}
        self.recordings = {}  # key -> EMGData

    def add_emg(self, key, emg_data):
        self.recordings[key] = emg_data

    def get_emg(self, key):
        return self.recordings[key]