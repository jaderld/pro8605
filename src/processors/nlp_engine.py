import re
import yaml
import os
import time
import logging
from textblob import TextBlob
from src.monitoring.metrics import FILLER_WORDS_COUNT, SENTIMENT_GAUGE, PROCESSING_TIME

class NLPEngine:
    def __init__(self, config_path='config/settings.yaml'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Liste des fillers
        raw_fillers = self.config.get('nlp', {}).get('fillers', ["euh", "bah", "ben", "genre", "voilà", "en fait", "du coup", "enfin"])
        self.fillers_regex = [r"\b" + f + r"\b" for f in raw_fillers]

    def _load_config(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _preclean(self, text: str) -> str:
        """Nettoyage inspiré du TP pour normaliser le texte avant analyse"""
        t = text.lower()
        # Supprimer la ponctuation collée
        t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
        # Compacter les espaces
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def analyze_text(self, text):
        start_time = time.time()
        if not text or not isinstance(text, str):
            return self._empty_result()

        try:
            # Nettoyage
            text_clean = self._preclean(text)
            
            results = {
                'filler_count': 0,
                'fillers_details': {},
                'sentiment_score': 0.0,
                'word_count': len(text_clean.split())
            }

            # 1. Détection des Tics
            for regex_pattern in self.fillers_regex:
                raw_word = regex_pattern.replace(r"\b", "")
                matches = len(re.findall(regex_pattern, text_clean))
                
                if matches > 0:
                    results['filler_count'] += matches
                    results['fillers_details'][raw_word] = matches

            # 2. Monitoring Prometheus (Sécurisé : sans labels pour correspondre à ton metrics.py)
            try:
                FILLER_WORDS_COUNT.inc(results['filler_count'])
            except: pass

            # 3. Analyse de Sentiment
            blob = TextBlob(text_clean)
            sentiment = blob.sentiment.polarity
            results['sentiment_score'] = round(sentiment, 2)
            
            try:
                SENTIMENT_GAUGE.set(sentiment)
                PROCESSING_TIME.labels(module='nlp').observe(time.time() - start_time)
            except: pass

            return results

        except Exception as e:
            self.logger.error(f"[NLPEngine] Error: {e}")
            return self._empty_result()

    def _empty_result(self):
        return {'filler_count': 0, 'fillers_details': {}, 'sentiment_score': 0.0, 'word_count': 0}