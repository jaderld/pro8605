import os
import sys

# Ajout du dossier racine au Python Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.dl_model import InterviewModel

def train_dl_only():
    print("==================================================")
    print("🫀 ENTRAÎNEMENT DU RÉSEAU DE NEURONES AUDIO (DL)")
    print("==================================================")

    try:
        dl_model = InterviewModel()
        
        print("⏳ Préparation en cours...")
        print("⚠️ La fonction d'entraînement PyTorch n'est pas encore implémentée dans dl_model.py.")
        print("💡 Prochaine étape : coder la fonction 'train_custom_model'.")
        
        # Quand tu auras codé la fonction, tu décommenteras ces lignes :
        # X_train_audio, y_train_audio = charger_les_vrais_audios()
        # dl_model.train_custom_model(X_train_audio, y_train_audio)
        # dl_model.save_custom_model('storage/models/emotion_net.pth')
        
    except Exception as e:
        print(f"❌ Erreur DL: {e}")

if __name__ == "__main__":
    train_dl_only()