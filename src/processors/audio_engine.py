import librosa
import numpy as np
import yaml
import os

class AudioEngine:
    def __init__(self, config_path='config/settings.yaml'):
        self.config = self._load_config(config_path)
        self.sample_rate = self.config['audio']['sample_rate']
        self.silence_threshold_db = self.config['audio']['silence_threshold_db']
        self.thresholds = self.config['thresholds']

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def process_signal(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            intervals = librosa.effects.split(y, top_db=self.silence_threshold_db)
            voiced = sum([(e-s) for s, e in intervals])
            pause_ratio = 1 - (voiced / len(y)) if len(y) > 0 else 0
            return {
                'duration': float(duration),
                'mean_volume': float(rms),
                'tempo': float(tempo),
                'pause_ratio': float(pause_ratio)
            }
        except Exception as e:
            print(f"[AudioEngine] Error processing signal: {e}")
            return {
                'duration': None,
                'mean_volume': None,
                'tempo': None,
                'pause_ratio': None
            }
