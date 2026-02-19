import librosa
import numpy as np
import yaml
import time
import logging
import os
import torch

from src.monitoring.metrics import PROCESSING_TIME, AUDIO_FEATURES_GAUGE

class AudioEngine:
    def __init__(self, config_path='config/settings.yaml'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Whisper et Silero travaillent de manière optimale à 16000 Hz
        self.sample_rate = 16000 
        
        # --- UPGRADE 1: Chargement de Silero-VAD ---
        self.logger.info("Chargement du modèle Silero-VAD...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        self.vad_model.eval()

    def _load_config(self, path):
        if not os.path.exists(path):
            self.logger.warning(f"Config file not found at {path}, using defaults.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def process_signal(self, file_path):
        start_time = time.time()
        try:
            # 1. Chargement de l'audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)

            # 2. Extraction Features de base (librosa)
            rms = np.mean(librosa.feature.rms(y=y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            # --- UPGRADE 2: Segmentation avec Silero-VAD ---
            wav_tensor = torch.from_numpy(y).float()
            speech_ts = self.get_speech_timestamps(wav_tensor, self.vad_model, sampling_rate=sr)
            
            non_silent_duration = 0.0
            segments = []
            for seg in speech_ts:
                start_s = seg['start'] / sr
                end_s = seg['end'] / sr
                non_silent_duration += (end_s - start_s)
                segments.append({"start_s": start_s, "end_s": end_s})

            pause_ratio = 1.0 - (non_silent_duration / duration) if duration > 0 else 0.0

            # 3. Envoi au Monitoring
            AUDIO_FEATURES_GAUGE.labels(feature='volume').set(rms)
            AUDIO_FEATURES_GAUGE.labels(feature='bpm').set(tempo)
            AUDIO_FEATURES_GAUGE.labels(feature='pause_ratio').set(pause_ratio)
            PROCESSING_TIME.labels(module='audio').observe(time.time() - start_time)

            features_vector = np.array([
                rms, zcr, spec_cent / 1000.0, tempo / 200.0, pause_ratio
            ], dtype=np.float32)

            return {
                'status': 'success',
                'meta': {'duration': round(float(duration), 2), 'sample_rate': sr},
                'features': {
                    'mean_volume': float(rms),
                    'tempo': float(tempo),
                    'pause_ratio': round(float(pause_ratio), 2),
                },
                'speech_segments': segments, # Transmis à dl_model.py !
                'dl_input_vector': features_vector
            }

        except Exception as e:
            self.logger.error(f"[AudioEngine] Error: {e}")
            return {
                'status': 'error', 'error': str(e),
                'dl_input_vector': np.zeros(5, dtype=np.float32)
            }