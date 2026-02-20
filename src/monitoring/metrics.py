from prometheus_client import Counter, Histogram, Gauge

# ==========================================
# 📊 PARTIE 1 : MÉTRIQUES "MÉTIER" (Pour les RH)
# (Ce que tu utilises déjà dans ton code actuel)
# ==========================================

PROCESSING_TIME = Histogram(
    'api_processing_time_seconds', 
    'Temps passé à traiter la requête',
    ['module'] 
)

TRANSCRIPTION_TIME = Histogram(
    'dl_transcription_time_seconds',
    'Temps passé par Whisper pour transcrire l\'audio'
)

FILLER_WORDS_COUNT = Counter(
    'nlp_filler_words_total',
    'Nombre total de tics de langage détectés'
)

SENTIMENT_GAUGE = Gauge(
    'nlp_sentiment_score',
    'Score de sentiment du texte (de -1 à 1)'
)

AUDIO_STRESS_LEVEL = Gauge(
    'audio_stress_level',
    'Niveau de stress calculé (0 = Calme, 1 = Stress maximum)'
)

AUDIO_FEATURES_GAUGE = Gauge(
    'audio_features',
    'Caractéristiques brutes extraites de l\'audio',
    ['feature'] 
)

FINAL_SCORE_GAUGE = Gauge(
    'interview_final_score',
    'Score final attribué au candidat (sur 100)'
)

# ==========================================
# ⚙️ PARTIE 2 : MÉTRIQUES "MLOps" (Pour le suivi technique)
# (Les nouvelles métriques à ajouter doucement)
# ==========================================

INFERENCE_TIME = Histogram(
    'model_inference_time_seconds',
    'Temps d\'exécution par modèle IA',
    ['model_name'], # 'whisper', 'pytorch_emotion', 'rf_scoring'
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")]
)

API_REQUESTS = Counter(
    'api_requests_total',
    'Nombre total de prédictions demandées',
    ['endpoint', 'status'] # 'success' ou 'error'
)

MODEL_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Niveau de confiance des prédictions PyTorch (de 0 à 1)',
    buckets=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)