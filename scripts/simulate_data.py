import os
import random
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Configuration
OUTPUT_FILE = 'storage/fake_sessions.csv'
NUM_SESSIONS = 2000 

def _score_filler_rate(filler_rate: float) -> float:
    """Pénalité progressive (quadratique) : chaque % de tics supplémentaire
    coûte plus cher que le précédent.
    0%→100, 3%→94, 5%→82, 8%→55, 10%→30, 12%→0.
    Basé sur : un orateur professionnel vise < 5 % de mots parasites.
    """
    fr_pct = filler_rate * 100.0
    return max(0.0, 100.0 - fr_pct ** 2 * 0.7)


def _score_tempo(wpm: float) -> float:
    """Zone optimale 110-170 mots/min (parole professionnelle française).
    En-dessous → manque d'énergie/hésitations, au-dessus → débit haché difficile à suivre.
    Ref : orateurs professionnels 120-150 WPM, journalistes 150-170 WPM.
    """
    if 110 <= wpm <= 170:
        return 100.0
    elif wpm < 110:
        return max(0.0, 100.0 - (110.0 - wpm) * 1.2)
    else:
        return max(0.0, 100.0 - (wpm - 170.0) * 1.2)


def _score_volume(vol_rms: float) -> float:
    """Voix bien projetée : zone optimale 25-70% (échelle ×1000).
    Trop faible = inaudible, trop fort = agressif.
    """
    vol_pct = vol_rms * 1000.0
    if 25.0 <= vol_pct <= 70.0:
        return 100.0
    elif vol_pct < 25.0:
        return vol_pct * 4.0
    else:
        return max(0.0, 100.0 - (vol_pct - 70.0) * 2.0)


def _score_pauses(pause_ratio: float) -> float:
    """Zone optimale 8-25% de pauses : structure et respiration du discours.
    Trop peu (<5%) = débit haché sans ponctuation, trop (>35%) = hésitation.
    Ref : les pauses intentionnelles améliorent la compréhension de 20%.
    """
    pct = pause_ratio * 100.0
    if 8.0 <= pct <= 25.0:
        return 100.0
    elif pct < 8.0:
        return pct * 12.0
    else:
        return max(0.0, 100.0 - (pct - 25.0) * 2.5)


def _score_richness(word_count: int) -> float:
    """Richesse de la réponse calibrée pour des entretiens de 2-10 minutes.
    < 100 mots = réponses trop courtes, > 500 mots = développé, > 800 = très élaboré.
    """
    if word_count >= 800:
        return 100.0
    elif word_count >= 500:
        return 85.0
    elif word_count >= 300:
        return 65.0
    elif word_count >= 150:
        return 45.0
    else:
        return 20.0


def generate_session(index):
    # ── 1. Paramètres sémantiques ─────────────────────────────────────────
    sentiment = random.uniform(-0.8, 0.8)

    # Durée plus longue : entretiens réalistes entre 2 et 10 minutes
    duration = round(random.uniform(120, 600), 2)

    # Débit : ~2.2 mots/s en parole française spontanée
    word_count = max(20, int(duration * random.gauss(2.2, 0.3)))

    # filler_rate tiré DIRECTEMENT depuis une distribution réaliste :
    # distribution exponentielle → majorité des candidats < 10%,
    # mais quelques cas pathologiques jusqu'à 40%.
    # Moyenne empirique recruteurs : ~5-8% de tics dans un entretien.
    filler_rate = float(np.clip(np.random.exponential(scale=0.07), 0.0, 0.40))
    filler_count = max(0, round(filler_rate * word_count))

    # ── 2. Label stress : probabilité floue (chevauchement volontaire) ────
    prob_stress = 0.5
    prob_stress += filler_rate * 0.5    # taux élevé de tics → plus de stress
    prob_stress -= sentiment * 0.2      # sentiment positif → moins de stress
    prob_stress = max(0.1, min(0.9, prob_stress))
    label = 1 if random.random() < prob_stress else 0

    # ── 3. Features audio (distribution gaussienne avec chevauchement) ────
    # WPM réel : word_count / (duration / 60), + variation selon le stress
    # Stressé → tend à parler plus vite (nervosité) ou à hésiter (lenteur)
    # On simule les deux cas avec une variance plus large pour le stressé
    base_wpm = (word_count / duration) * 60  # ~132 WPM en moyenne

    if label == 1:  # STRESSÉ
        volume     = random.gauss(0.05, 0.03)
        wpm        = base_wpm * random.gauss(1.10, 0.18)  # plus rapide ou erratique
        zcr        = random.gauss(0.10, 0.05)
        spec_cent  = random.gauss(2.2, 0.7)
        pause_ratio = random.gauss(0.32, 0.14)
    else:           # CALME
        volume     = random.gauss(0.07, 0.04)
        wpm        = base_wpm * random.gauss(1.00, 0.09)  # proche du naturel
        zcr        = random.gauss(0.07, 0.04)
        spec_cent  = random.gauss(1.8, 0.6)
        pause_ratio = random.gauss(0.16, 0.09)

    # Erreur d'étiquetage humaine (10%)
    if random.random() < 0.10:
        label = 1 - label

    # Clamping
    volume      = max(0.01, min(0.15,  volume))
    wpm         = max(60.0, min(280.0, wpm))
    zcr         = max(0.02, min(0.20,  zcr))
    spec_cent   = max(0.8,  min(3.5,   spec_cent))
    pause_ratio = max(0.03, min(0.60,  pause_ratio))

    # ── 4. Score cible : formule pondérée ancrée sur les critères RH ─────
    # Poids inspirés des grilles d'évaluation en communication orale :
    #   35% tics de langage (critère n°1 des recruteurs — pénalité quadratique)
    #   18% débit vocal    (énergie, dynamisme)
    #   14% sentiment      (positivité, engagement)
    #   10% volume         (projection, assertivité)
    #   10% pauses         (structure, maîtrise)
    #    8% émotion/stress (contrôle de soi)
    #    5% richesse       (développement des réponses)
    raw_score = (
        _score_filler_rate(filler_rate) * 0.35
        + _score_tempo(wpm)             * 0.18
        + ((sentiment + 1) * 50)        * 0.14   # -1→0, 0→50, +1→100
        + _score_volume(volume)         * 0.10
        + _score_pauses(pause_ratio)    * 0.10
        + (30.0 if label == 1 else 80.0)* 0.08   # stressé pénalisé
        + _score_richness(word_count)   * 0.05
    )

    # Bruit réaliste ±5 pts (variabilité inter-évaluateurs humaine)
    noise = random.gauss(0, 5)
    final_score = max(0, min(100, int(round(raw_score + noise))))

    return {
        "session_id":        f"sess_{index}_{int(time.time())}",
        "timestamp":         datetime.now().isoformat(),
        "duration":          duration,
        "volume":            round(volume, 4),
        "tempo":             round(wpm, 1),
        "pause_ratio":       round(pause_ratio, 3),
        "zcr":               round(zcr, 4),
        "spectral_centroid": round(spec_cent, 4),
        "sentiment":         round(sentiment, 3),
        "filler_count":      filler_count,
        "word_count":        word_count,
        "filler_rate":       round(filler_rate, 4),
        "label":             label,
        "stress_level":      float(label),
        "target_score":      final_score,
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