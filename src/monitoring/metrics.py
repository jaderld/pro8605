from prometheus_client import Counter, Histogram, Gauge

# --- 1. Métriques de Performance (Temps) ---
PROCESSING_TIME = Histogram(
    'api_processing_time_seconds', 
    'Temps passé à traiter la requête',
    ['module'] 
)

TRANSCRIPTION_TIME = Histogram(
    'dl_transcription_time_seconds',
    'Temps passé par Whisper pour transcrire l\'audio'
)

# --- 2. Métriques NLP (Texte) ---
FILLER_WORDS_COUNT = Counter(
    'nlp_filler_words_total',
    'Nombre total de tics de langage détectés'
)

SENTIMENT_GAUGE = Gauge(
    'nlp_sentiment_score',
    'Score de sentiment du texte (de -1 à 1)'
)

# --- 3. Métriques Audio & Émotion ---
AUDIO_STRESS_LEVEL = Gauge(
    'audio_stress_level',
    'Niveau de stress calculé (0 = Calme, 1 = Stress maximum)'
)

AUDIO_FEATURES_GAUGE = Gauge(
    'audio_features',
    'Caractéristiques brutes extraites de l\'audio',
    ['feature'] # Permet de séparer 'tempo', 'volume', 'pause_ratio'
)

# --- 4. Métrique ML (Score Final) ---
# Au cas où ton ml_model.py ou api/main.py tente de l'exporter
FINAL_SCORE_GAUGE = Gauge(
    'interview_final_score',
    'Score final attribué au candidat (sur 100)'
)