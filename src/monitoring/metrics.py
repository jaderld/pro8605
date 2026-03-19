from prometheus_client import Counter, Histogram, Gauge

# ==========================================
# ÉTAPE 1 — AUDIO ENGINE
# ==========================================

PROCESSING_TIME = Histogram(
    'api_processing_time_seconds',
    'Temps de traitement par module du pipeline',
    ['module']
)

AUDIO_FEATURES_GAUGE = Gauge(
    'audio_features',
    'Caractéristiques brutes extraites de l\'audio',
    ['feature']
)

AUDIO_DURATION = Histogram(
    'audio_duration_seconds',
    'Durée des fichiers audio analysés',
    buckets=[1, 3, 5, 10, 20, 30, 60, float("inf")]
)

AUDIO_PAUSE_RATIO = Gauge(
    'audio_pause_ratio',
    'Ratio de pauses (0 = pas de pauses, 1 = que des pauses)'
)

AUDIO_TEMPO = Gauge(
    'audio_tempo_bpm',
    'Tempo détecté en BPM'
)

AUDIO_VOLUME = Gauge(
    'audio_volume_rms',
    'Volume RMS moyen de l\'audio'
)

# ==========================================
# ÉTAPE 2 — TRANSCRIPTION (Whisper)
# ==========================================

TRANSCRIPTION_TIME = Histogram(
    'dl_transcription_time_seconds',
    'Temps de transcription Whisper',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
)

TRANSCRIPTION_WORD_COUNT = Histogram(
    'transcription_word_count',
    'Nombre de mots dans la transcription',
    buckets=[5, 10, 20, 40, 80, 150, float("inf")]
)

# ==========================================
# ÉTAPE 3 — NLP ENGINE
# ==========================================

FILLER_WORDS_COUNT = Counter(
    'nlp_filler_words_total',
    'Nombre total de tics de langage détectés'
)

FILLER_WORDS_PER_ANALYSIS = Histogram(
    'nlp_filler_words_per_analysis',
    'Nombre de tics par analyse',
    buckets=[0, 1, 2, 3, 5, 8, 12, float("inf")]
)

SENTIMENT_GAUGE = Gauge(
    'nlp_sentiment_score',
    'Score de sentiment du texte (de -1 à 1)'
)

# ==========================================
# ÉTAPE 4 — MODÈLE DL (Émotion)
# ==========================================

AUDIO_STRESS_LEVEL = Gauge(
    'audio_stress_level',
    'Niveau de stress détecté (0 = Calme, 1 = Stressé)'
)

INFERENCE_TIME = Histogram(
    'model_inference_time_seconds',
    'Temps d\'inférence par modèle IA',
    ['model_name'],
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")]
)

MODEL_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Confiance des prédictions du modèle d\'émotion (0 à 1)',
    buckets=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# ==========================================
# ÉTAPE 5 — SCORING FINAL (Random Forest)
# ==========================================

FINAL_SCORE_GAUGE = Gauge(
    'interview_final_score',
    'Score final attribué au candidat (sur 100)'
)

BASE_SCORE_GAUGE = Gauge(
    'interview_base_score',
    'Score brut Random Forest avant pénalités'
)

SCORE_PENALTY = Histogram(
    'interview_score_penalty',
    'Total des pénalités appliquées (tics + sentiment)',
    buckets=[0, 5, 10, 20, 30, 50, 80, float("inf")]
)

SCORE_INTERPRETATION = Counter(
    'interview_score_interpretation_total',
    'Distribution des interprétations de score',
    ['interpretation']  # 'Excellent', 'Moyen', 'À améliorer'
)

# ==========================================
# MONITORING API
# ==========================================

API_REQUESTS = Counter(
    'api_requests_total',
    'Nombre total de requêtes API',
    ['endpoint', 'status']
)