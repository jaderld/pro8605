import os
import random
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Configuration
OUTPUT_FILE = 'storage/fake_sessions.csv'
NUM_SESSIONS = 2000 

def generate_session(index):
    # 1. On définit un profil de base (Sémantique)
    # On tire un sentiment et des tics de manière décorrélée au début
    sentiment = random.uniform(-0.8, 0.8)
    filler_count = random.randint(0, 8)
    
    # 2. LA LOGIQUE DE DÉCISION DU LABEL (BEAUCOUP PLUS FLOU)
    # Au lieu d'une règle, on crée un "Score de Probabilité de Stress"
    prob_stress = 0.5 # On part de 50/50
    prob_stress += (filler_count * 0.05)   # Plus de tics augmente la proba
    prob_stress -= (sentiment * 0.2)      # Un bon sentiment la diminue
    
    # On sature la proba entre 0.1 et 0.9 pour qu'il y ait TOUJOURS un doute
    prob_stress = max(0.1, min(0.9, prob_stress))
    
    # On détermine le label
    label = 1 if random.random() < prob_stress else 0

    # 3. GÉNÉRATION DES FEATURES AVEC UN FORT CHEVAUCHEMENT (OVERLAP)
    # Ici, on rapproche les moyennes et on augmente l'écart-type
    if label == 1: # STRESSÉ
        volume = random.gauss(0.05, 0.03) 
        bpm = random.gauss(130, 25)      # Très large, peut descendre à 100
        zcr = random.gauss(0.10, 0.05)   # Très proche du calme
        spec_cent = random.gauss(2.2, 0.7)
        pause_ratio = random.gauss(0.35, 0.15)
    else: # CALME
        volume = random.gauss(0.07, 0.04) 
        bpm = random.gauss(110, 20)      # Très large, peut monter à 130
        zcr = random.gauss(0.07, 0.04)   # On chevauche volontairement le 0.10 du stress
        spec_cent = random.gauss(1.7, 0.6)
        pause_ratio = random.gauss(0.18, 0.10)

    # 4. LE COUP DE GRÂCE : L'ERREUR D'ÉTIQUETAGE (Label Noise)
    # Dans 12% des cas, on inverse le label sans changer les caractéristiques.
    # C'est ce qui simule l'erreur humaine (un recruteur qui se trompe).
    if random.random() < 0.12:
        label = 1 - label

    # Clamping (Nettoyage des valeurs extrêmes)
    volume = max(0.01, min(0.15, volume))
    zcr = max(0.02, min(0.20, zcr))
    spec_cent = max(0.8, min(3.5, spec_cent))
    pause_ratio = max(0.05, min(0.6, pause_ratio))

    # 5. SCORE FINAL (Random Forest) - On ajoute beaucoup de bruit
    noise = random.uniform(-12, 12) # +/- 12 points de pur hasard
    score = 85 - (pause_ratio * 60) - (filler_count * 3) + (sentiment * 10) - (label * 5) + noise
    final_score = max(0, min(100, int(score)))

    return {
        "session_id": f"sess_{index}_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "duration": round(random.uniform(30, 300), 2),
        "volume": round(volume, 4),
        "tempo": round(bpm, 1),
        "pause_ratio": round(pause_ratio, 2),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(spec_cent, 4),
        "sentiment": round(sentiment, 2),
        "filler_count": filler_count,
        "label": label,
        "stress_level": float(label),
        "target_score": final_score
    }

def main():
    print(f"🚀 Génération de {NUM_SESSIONS} sessions ultra-bruitées...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data = [generate_session(i) for i in range(NUM_SESSIONS)]
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Terminé. Le modèle va enfin devoir 'réfléchir'.")

if __name__ == "__main__":
    main()