# Placeholder for future evaluation metrics
# You can add more advanced metrics here as needed

def compute_soft_skills(audio_metrics, nlp_metrics):
    """
    Calcule trois scores de soft skills indépendants à partir des métriques audio et NLP.
    Chaque score utilise des sources de données différentes pour éviter la redondance.
    """
    def clamp(v): return max(0.0, min(1.0, v))

    pause_ratio  = audio_metrics.get('pause_ratio', 0.0)
    sentiment    = nlp_metrics.get('sentiment_score', 0.0)      # [-1, 1]
    tempo        = audio_metrics.get('tempo', 0.0)              # BPM
    volume       = audio_metrics.get('volume', 0.0)             # RMS
    filler_count = nlp_metrics.get('filler_count', 0)

    # Stress : augmente avec les pauses, le sentiment négatif et les tics de langage
    stress = clamp(
        pause_ratio * 0.4
        + (1 - (sentiment + 1) / 2) * 0.3          # sentiment normalisé [0,1] inversé
        + min(filler_count / 10.0, 1.0) * 0.3
    )

    # Confidence : augmente avec le sentiment positif, le volume et peu de pauses
    confidence = clamp(
        ((sentiment + 1) / 2) * 0.4                # sentiment normalisé [0,1]
        + min(volume / 0.1, 1.0) * 0.3
        + (1 - pause_ratio) * 0.3
    )

    # Dynamisme : débit + énergie vocale (indépendant du contenu)
    dynamism = clamp(
        min(tempo / 130.0, 1.0) * 0.6
        + min(volume / 0.1, 1.0) * 0.4
    )

    return {
        'stress': round(stress, 4),
        'confidence': round(confidence, 4),
        'dynamism': round(dynamism, 4),
    }
