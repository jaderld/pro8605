import os
import sys
import pandas as pd

# Ajout du dossier racine au Python Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.ml_model import ScoringModel

def train_ml_only():
    print("==================================================")
    print("🚀 ENTRAÎNEMENT DU MODÈLE DE SCORING (ML)")
    print("==================================================")

    try:
        csv_path = os.path.join(parent_dir, 'storage', 'fake_sessions.csv')
        
        if not os.path.exists(csv_path):
            print(f"❌ Erreur : Le fichier {csv_path} est introuvable.")
            return

        df = pd.read_csv(csv_path)
        
        ml_model = ScoringModel()
        ml_model.train(df) 
        
        print("\n✅ ENTRAÎNEMENT ML TERMINÉ ! (Fichier .pkl mis à jour)")
        
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

if __name__ == "__main__":
    train_ml_only()