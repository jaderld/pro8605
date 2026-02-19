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

# --- FONCTION PRECLEAN (Inspirée du TP) ---
def preclean(text: str) -> str:
    """Nettoie les erreurs classiques de l'ASR avant le passage au NLP."""
    t = text.lower()
    # Séparer chiffres collés à des lettres
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    # Espace après la ponctuation
    t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
    # Compacter les espaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

class SimpleAudioNet(nn.Module):
    # ... (Garde exactement ton code actuel pour SimpleAudioNet) ...
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

    def transcribe_audio(self, audio_path, speech_segments=None):
        """
        UPGRADE 3 & 4 : Transcription Segmentée & RTF
        """
        start_time = time.time()
        
        # On charge l'audio brut
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration_s = len(audio) / sr
        full_text = ""

        # Si le VAD a trouvé des segments, on ne transcrit que ceux-là
        if speech_segments and len(speech_segments) > 0:
            print(f"Transcription segmentée : {len(speech_segments)} segments détectés.")
            for seg in speech_segments:
                start_sample = int(seg['start_s'] * sr)
                end_sample = int(seg['end_s'] * sr)
                chunk = audio[start_sample:end_sample]
                
                # Inférence Whisper sur le bout d'audio
                res = self.transcriber.transcribe(chunk, fp16=False, language="fr")
                full_text += res["text"] + " "
        else:
            # Fallback : on transcrit tout si le VAD n'a pas été fourni
            print("Transcription intégrale (Pas de segments VAD fournis).")
            res = self.transcriber.transcribe(audio, fp16=False, language="fr")
            full_text = res["text"]

        # Nettoyage du texte généré
        cleaned_text = preclean(full_text)

        # Calcul du temps et du RTF
        elapsed_s = time.time() - start_time
        rtf = elapsed_s / max(audio_duration_s, 1e-9)
        print(f"⏱️ ASR Terminé. RTF: {rtf:.3f} (Audio: {audio_duration_s:.1f}s, Calcul: {elapsed_s:.1f}s)")
        
        # Enregistrement métrique Prometheus
        TRANSCRIPTION_TIME.observe(elapsed_s)
        
        return cleaned_text

    # ... (Garde tes méthodes predict_emotion, train_custom_model et save_custom_model intactes) ...
    def predict_emotion(self, features):
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        classes = {0: "Calme", 1: "Neutre", 2: "Stressé"}
        result_label = classes.get(predicted_class, "Inconnu")
        mlflow.log_metric("emotion_confidence", confidence)
        stress_val = 1.0 if result_label == "Stressé" else (0.5 if result_label == "Neutre" else 0.0)
        AUDIO_STRESS_LEVEL.set(stress_val)
        return {
            "emotion": result_label,
            "confidence": round(confidence, 4),
            "raw_logits": outputs.cpu().numpy().tolist()
        }