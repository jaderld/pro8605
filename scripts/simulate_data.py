import random
from datetime import datetime, timedelta
from database.db_manager import DBManager

FILLER_WORDS = ['euh', 'bah', 'genre', 'du coup']

def random_session():
    now = datetime.now()
    duration = round(random.uniform(60, 300), 2)
    sentiment = round(random.uniform(-0.5, 0.9), 2)
    pause_ratio = round(random.uniform(0.05, 0.4), 2)
    transcription = random.choice([
        "Bonjour, je m'appelle Jean et euh je postule pour le poste...",
        "Alors bah du coup j'ai travaillé chez XYZ...",
        "Je suis très motivé et genre prêt à apprendre...",
        "Je pense que mes compétences correspondent..."
    ])
    # Générer un label pour ML (ex: 0 = stressé, 1 = confiant)
    label = 1 if sentiment > 0.2 and pause_ratio < 0.2 else 0
    # Générer un chemin audio fictif
    audio_path = f"audio/fake_audio_{random.randint(1,1000)}.wav"
    metrics = {
        'timestamp': (now - timedelta(days=random.randint(0, 30))).isoformat(),
        'duration': duration,
        'sentiment_score': sentiment,
        'pause_ratio': pause_ratio,
        'transcription': transcription,
        'audio_path': audio_path,
        'label': label,
        'full_metrics_json': '{}'
    }
    return metrics

def main():
    db = DBManager()
    sessions = []
    for _ in range(20):
        metrics = random_session()
        db.save_session(metrics)
        sessions.append(metrics)
    # Générer un CSV pour le pipeline ML
    import pandas as pd
    df = pd.DataFrame(sessions)
    df.to_csv('storage/fake_sessions.csv', index=False)
    print("20 fake sessions inserted et CSV généré.")

if __name__ == "__main__":
    main()
