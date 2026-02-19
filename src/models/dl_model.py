import torch
import torch.nn as nn
import torch.optim as optim
import whisper
import mlflow
import os
import time
import re
import librosa
import numpy as np
from src.monitoring.metrics import TRANSCRIPTION_TIME, AUDIO_STRESS_LEVEL

# --- FONCTION PRECLEAN ---
def preclean(text: str) -> str:
    """Nettoie les erreurs classiques de l'ASR avant le passage au NLP."""
    t = text.lower()
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

class SimpleAudioNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleAudioNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class InterviewModel:
    def __init__(self, model_path="storage/models/emotion_net.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Chargement de Whisper sur {self.device}...")
        self.transcriber = whisper.load_model("base", device=self.device)
        
        self.classifier = SimpleAudioNet(input_dim=5, num_classes=3).to(self.device)
        self.classifier.eval()
        
        if os.path.exists(model_path):
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Modèle d'émotion personnalisé chargé.")
        else:
            print("Aucun modèle d'émotion trouvé, utilisation des poids initiaux.")

    def transcribe_audio(self, audio_path):
        """
        Transcription intégrale AVEC conservation des tics de langage (euh, bah, voilà).
        """
        start_time = time.time()
        
        # 1. On charge l'audio complet (Whisper a besoin du contexte total)
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration_s = len(audio) / sr

        # 2. PROMPT MAGIQUE : On "apprend" à Whisper à ne pas censurer les hésitations
        tic_prompt = "C'est un entretien d'embauche. Le candidat hésite souvent, il dit euh, bah, voilà, du coup."

        print(f"🎙️ Whisper analyse l'audio complet ({audio_duration_s:.1f}s)...")
        
        # 3. Inférence Whisper
        # initial_prompt : la clé pour garder les 'euh'
        # language="fr" : évite qu'il traduise les tics en anglais
        res = self.transcriber.transcribe(
            audio, 
            fp16=False, 
            language="fr", 
            initial_prompt=tic_prompt
        )
        
        raw_text = res["text"]

        # 4. Nettoyage léger (Preclean)
        cleaned_text = preclean(raw_text)

        # Calcul du RTF (Real Time Factor)
        elapsed_s = time.time() - start_time
        rtf = elapsed_s / max(audio_duration_s, 1e-9)
        print(f"⏱️ ASR Terminé. RTF: {rtf:.3f} | Texte: {cleaned_text[:50]}...")
        
        TRANSCRIPTION_TIME.observe(elapsed_s)
        
        return cleaned_text

    def predict_emotion(self, features):
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        classes = {0: "Calme", 1: "Neutre", 2: "Stressé"}
        result_label = classes.get(predicted_class, "Inconnu")
        
        try:
            mlflow.log_metric("emotion_confidence", confidence)
        except:
            pass
            
        stress_val = 1.0 if result_label == "Stressé" else (0.5 if result_label == "Neutre" else 0.0)
        AUDIO_STRESS_LEVEL.set(stress_val)
        
        return {
            "emotion": result_label,
            "confidence": round(confidence, 4)
        }