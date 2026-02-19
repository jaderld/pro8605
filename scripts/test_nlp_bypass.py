import sys
import os
import pandas as pd

# --- 1. AJOUT DU DOSSIER RACINE AU PYTHON PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.processors.nlp_engine import NLPEngine
from src.models.ml_model import ScoringModel  # <-- Ajout du Modèle ML

def run_test():
    print("\n🚀 DÉMARRAGE DU TEST NLP + ML (SANS AUDIO)")
    print("=========================================")
    
    # --- 2. INITIALISATION DES MOTEURS ---
    try:
        config_path = os.path.join(parent_dir, 'config', 'settings.yaml')
        engine = NLPEngine(config_path=config_path)
        print("✅ Moteur NLP initialisé.")
    except Exception as e:
        engine = NLPEngine()
        print("⚠️ Utilisation de la configuration NLP par défaut.")

    # Chargement et entraînement forcé du modèle ML pour le test
    print("🧠 Entraînement du Juge (Random Forest) en cours...")
    ml_model = ScoringModel()
    csv_path = os.path.join(parent_dir, 'storage', 'fake_sessions.csv')
    df = pd.read_csv(csv_path)
    ml_model.train(df)
    print("✅ Moteur ML prêt à noter.")

    # --- 3. JEU DE DONNÉES DE TEST ---
    test_sentences = [
        "Bonjour, euh... je suis très content de, ben, passer cet entretien.",
        "Alors du coup, j'ai travaillé chez Google, voilà, c'était super.",
        "Je suis une phrase parfaite sans aucune hésitation. Tout est merveilleux !",
        "Euh... bah... je ne sais pas trop, genre, c'est compliqué quoi. Je déteste ça."
    ]

    # Faux paramètres audio pour simuler Whisper
    mock_audio_features = {'pause_ratio': 0.15}

    # --- 4. EXÉCUTION ---
    print(f"\n{'PHRASE (Tronquée)':<45} | {'TICS':<5} | {'SENTIMENT':<9} | {'SCORE'}")
    print("-" * 80)

    for text in test_sentences:
        # Analyse NLP
        result = engine.analyze_text(text)
        
        # Extraction résultats
        fillers = result.get('filler_count', 0)
        sentiment = result.get('sentiment_score', 0.0)
        details = result.get('fillers_details', {})
        
        # Le ML donne la note
        score = ml_model.predict_score(audio_features=mock_audio_features, nlp_results=result)
        
        # Affichage (Ton design)
        display_text = (text[:42] + '...') if len(text) > 42 else text
        print(f"{display_text:<45} | {fillers:<5} | {sentiment:<9} | 🏆 {score}/100")
        
        if fillers > 0:
            print(f"   L-> Détails: {details}")

    print("-" * 80)
    print("✅ TEST TERMINÉ. Observe les scores selon les TICS et le SENTIMENT !")

if __name__ == "__main__":
    run_test()