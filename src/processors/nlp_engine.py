import re
import yaml
import os
import time
import logging
from transformers import pipeline as hf_pipeline
from src.monitoring.metrics import (
    FILLER_WORDS_COUNT,
    FILLER_WORDS_PER_ANALYSIS,
    SENTIMENT_GAUGE,
    PROCESSING_TIME,
    TRANSCRIPTION_WORD_COUNT,
)

logger = logging.getLogger(__name__)

# Mapping des labels de sortie du modèle (1–5 étoiles) vers un score [-1, +1]
# Formule: score_i = (i - 3) / 2  →  1★=-1.0, 2★=-0.5, 3★=0.0, 4★=+0.5, 5★=+1.0
_STAR_WEIGHTS = {
    "1 star":  -1.0,
    "2 stars": -0.5,
    "3 stars":  0.0,
    "4 stars":  0.5,
    "5 stars":  1.0,
}


class NLPEngine:
    """
    Moteur d'analyse NLP pour entretiens en français.

    Fonctionnalités :
    - Détection des tics de langage (fillers) par regex
    - Analyse de sentiment via DistilCamemBERT (modèle natif français)
    - Métriques Prometheus émises à chaque analyse
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        raw_fillers = self.config.get(
            "fillers",
            ["euh", "bah", "ben", "genre", "voilà", "en fait", "du coup", "enfin"],
        )
        self.fillers_regex = [r"\b" + f + r"\b" for f in raw_fillers]

        self.logger.info("[NLPEngine] Chargement de DistilCamemBERT-sentiment...")
        self._sentiment_pipe = hf_pipeline(
            "text-classification",
            model="cmarkea/distilcamembert-base-sentiment",
            top_k=None,   # retourne les probabilités pour chacun des 5 labels
            device=-1,    # CPU uniquement (compatible Docker sans GPU)
        )
        self.logger.info("[NLPEngine] DistilCamemBERT prêt.")

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _preclean(self, text: str) -> str:
        """Normalisation du texte : minuscules, ponctuation, espaces."""
        t = text.lower()
        t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
        t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
        t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _compute_sentiment(self, text_clean: str) -> float:
        """
        Calcule un score de sentiment continu dans [-1, +1] à partir des
        probabilités retournées par DistilCamemBERT-sentiment.

        Le modèle prédit 5 classes (1 à 5 étoiles). On pondère chaque
        probabilité par le score correspondant et on somme.
        """
        # Le modèle est robuste jusqu'à 512 tokens ; on tronque en caractères
        # (approximation suffisante : 1 token ≈ 4 caractères en français)
        truncated = text_clean[:2048]
        raw = self._sentiment_pipe(truncated)  # [[{label, score}, ...]]
        star_probs = {r["label"]: r["score"] for r in raw[0]}
        score = sum(
            _STAR_WEIGHTS.get(label, 0.0) * prob
            for label, prob in star_probs.items()
        )
        return round(max(-1.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """
        Analyse complète du texte transcrit.

        Retourne :
            filler_count (int)      : nombre total de tics de langage
            fillers_details (dict)  : {filler: occurrences}
            sentiment_score (float) : score [-1, +1] (négatif → positif)
            word_count (int)        : nombre de mots
        """
        start_time = time.time()

        if not text or not isinstance(text, str):
            return self._empty_result()

        try:
            text_clean = self._preclean(text)
            word_count = len(text_clean.split())

            # --- 1. Détection des tics de langage ---
            filler_count = 0
            fillers_details: dict = {}
            for pattern in self.fillers_regex:
                word = pattern.replace(r"\b", "")
                n = len(re.findall(pattern, text_clean))
                if n > 0:
                    filler_count += n
                    fillers_details[word] = n

            # --- 2. Analyse de sentiment (DistilCamemBERT) ---
            sentiment_score = self._compute_sentiment(text_clean)

            # --- 3. Métriques Prometheus ---
            FILLER_WORDS_COUNT.inc(filler_count)
            FILLER_WORDS_PER_ANALYSIS.observe(filler_count)
            TRANSCRIPTION_WORD_COUNT.observe(word_count)
            SENTIMENT_GAUGE.set(sentiment_score)
            PROCESSING_TIME.labels(module="nlp").observe(time.time() - start_time)

            return {
                "filler_count": filler_count,
                "fillers_details": fillers_details,
                "sentiment_score": sentiment_score,
                "word_count": word_count,
            }

        except Exception as e:
            self.logger.error(f"[NLPEngine] Erreur lors de l'analyse : {e}", exc_info=True)
            return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "filler_count": 0,
            "fillers_details": {},
            "sentiment_score": 0.0,
            "word_count": 0,
        }