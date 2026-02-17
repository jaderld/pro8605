import os
import yaml
import torch
import whisper
from textblob import TextBlob

class NLPEngine:
    _whisper_model = None

    def __init__(self, config_path='config/settings.yaml'):
        self.config = self._load_config(config_path)
        self.fillers = self.config['fillers']

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @classmethod
    def get_whisper_model(cls):
        if cls._whisper_model is None:
            cls._whisper_model = whisper.load_model('base')
        return cls._whisper_model

    def process_text(self, file_path):
        try:
            model = self.get_whisper_model()
            result = model.transcribe(file_path, language='fr')
            transcription = result['text']
            lang = result.get('language', 'fr')
            blob = TextBlob(transcription)
            sentiment = blob.sentiment.polarity
            filler_count = sum(transcription.lower().count(f) for f in self.fillers)
            return {
                'transcription': transcription,
                'language': lang,
                'sentiment_score': float(sentiment),
                'filler_count': int(filler_count)
            }
        except Exception as e:
            print(f"[NLPEngine] Error processing text: {e}")
            return {
                'transcription': None,
                'language': None,
                'sentiment_score': None,
                'filler_count': None
            }
