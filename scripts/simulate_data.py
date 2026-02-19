import os
import random
import pandas as pd
import time
from datetime import datetime, timedelta

# Configuration
OUTPUT_FILE = 'storage/fake_sessions.csv'
NUM_SESSIONS = 500  # On génère 500 entretiens pour avoir de la data

# Données fictives pour la simulation
TEXT_SAMPLES = [
    ("Bonjour, je suis très motivé pour ce poste.", 0.8, 0),
    ("Euh... bah en fait je sais pas trop, genre...", -0.2, 5),
    ("J'ai une expérience solide en Python et Docker.", 0.6, 0),
    ("C'était un peu compliqué, euh, mais on a réussi.", 0.1, 2),
    ("Franchement c'était nul, je déteste le code.", -0.9, 0),
    ("Euh... alors... je crois que... peut-être.", -0.1, 8)
]

def generate_session(index):
    """
    Génère une session d'entretien avec des métriques cohérentes.
    """
    # 1. Choix d'un profil de texte de base
    base_text, base_sentiment, base_fillers = random.choice(TEXT_SAMPLES)
    
    # 2. Ajout de variation aléatoire (Bruit)
    sentiment = min(1.0, max(-1.0, base_sentiment + random.uniform(-0.2, 0.2)))
    
    # Le nombre de fillers varie autour de la base
    filler_count = max(0, base_fillers + random.randint(-1, 3))
    
    # 3. Génération des métriques Audio (corrélées au stress)
    # Si beaucoup de fillers -> souvent plus de pauses et débit (bpm) instable
    is_stressed = filler_count > 3
    
    if is_stressed:
        pause_ratio = random.uniform(0.2, 0.5)  # Beaucoup de silence
        bpm = random.uniform(110, 160)          # Cœur qui bat vite ou débit rapide
        volume = random.uniform(0.01, 0.05)     # Parle doucement (timide)
    else:
        pause_ratio = random.uniform(0.05, 0.15) # Fluide
        bpm = random.uniform(90, 120)            # Calme
        volume = random.uniform(0.05, 0.15)      # Voix posée

    duration = random.uniform(30, 300)

    # 4. CALCUL DU LABEL (La note que l'IA doit apprendre à prédire)
    # Formule : 100 pts de base 
    # - pénalité pauses 
    # - pénalité fillers 
    # + bonus sentiment
    score = 100 - (pause_ratio * 100) - (filler_count * 5) + (sentiment * 20)
    
    # Ajout d'un peu d'aléatoire pour ne pas que ce soit trop mathématique
    score += random.uniform(-5, 5)
    
    # Bornage entre 0 et 100
    final_score = max(0, min(100, int(score)))

    # Classification binaire pour certains modèles (0 = Rejeté, 1 = Accepté)
    label = 1 if final_score > 60 else 0

    return {
        "session_id": f"sess_{index}_{int(time.time())}",
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat(),
        "duration": round(duration, 2),
        "mean_volume": round(volume, 4),
        "tempo": round(bpm, 1),
        "pause_ratio": round(pause_ratio, 2),
        "sentiment": round(sentiment, 2),
        "filler_count": filler_count,
        "transcription_sample": base_text,
        "target_score": final_score, # Pour la régression (Note /100)
        "label": label               # Pour la classification (Oui/Non)
    }

def main():
    print(f"🚀 Génération de {NUM_SESSIONS} sessions d'entraînement...")
    
    # Création du dossier storage s'il n'existe pas
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    data = []
    for i in range(NUM_SESSIONS):
        data.append(generate_session(i))
    
    # Conversion en DataFrame Pandas
    df = pd.DataFrame(data)
    
    # Sauvegarde CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Terminé ! Fichier sauvegardé : {OUTPUT_FILE}")
    print("📊 Aperçu des données :")
    print(df[['filler_count', 'pause_ratio', 'target_score', 'label']].head())

if __name__ == "__main__":
    main()