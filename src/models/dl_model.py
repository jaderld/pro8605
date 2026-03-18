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
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.monitoring.metrics import TRANSCRIPTION_TIME, AUDIO_STRESS_LEVEL, INFERENCE_TIME, MODEL_CONFIDENCE

# --- FONCTION PRECLEAN ---
def preclean(text: str) -> str:
    """Nettoyage normalisant pour éviter les erreurs de parsing du moteur NLP."""
    t = text.lower()
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# --- ARCHITECTURE DU RÉSEAU DE NEURONES ---
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

# --- CLASSE PRINCIPALE DU MODÈLE ---
class InterviewModel:
    def __init__(self, model_path="storage/models/emotion_net.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Chargement de Whisper sur {self.device}...")
        self.transcriber = whisper.load_model("base", device=self.device)
        
        # Configuration Binaire : 2 classes (Calme = 0, Stressé = 1)
        self.classifier = SimpleAudioNet(input_dim=5, num_classes=2).to(self.device)
        self.classifier.eval()
        
        if os.path.exists(model_path):
            try:
                self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✅ Modèle d'émotion personnalisé (2 classes) chargé.")
            except Exception as e:
                print(f"⚠️ Erreur : {e}. Architecture incompatible ou fichier corrompu.")
        else:
            print("ℹ️ Aucun modèle d'émotion trouvé, utilisation des poids par défaut.")

    def transcribe_audio(self, audio_path):
        """Transcription via Whisper avec Prompt Engineering pour les tics."""
        start_time = time.time()
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration_s = len(audio) / sr

        # Prompt pour forcer Whisper à transcrire les hésitations sans les censurer
        tic_prompt = "C'est un entretien d'embauche. Le candidat hésite souvent, il dit euh, bah, voilà, du coup."

        res = self.transcriber.transcribe(
            audio, 
            fp16=False, 
            language="fr", 
            initial_prompt=tic_prompt
        )
        
        cleaned_text = preclean(res["text"])
        
        elapsed_s = time.time() - start_time
        TRANSCRIPTION_TIME.observe(elapsed_s)
        
        return cleaned_text

    def predict_emotion(self, features):
        """Inférence PyTorch pour la classification d'émotion."""
        start_time = time.time()
        self.classifier.eval()
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        elapsed = time.time() - start_time
        INFERENCE_TIME.labels(model_name='pytorch_emotion').observe(elapsed)
        MODEL_CONFIDENCE.observe(confidence)

        classes = {0: "Calme", 1: "Stressé"}
        result_label = classes.get(predicted_class, "Inconnu")
        
        # Monitoring MLflow et Prometheus
        try:
            mlflow.log_metric("emotion_confidence", confidence)
        except: pass
            
        stress_val = 1.0 if result_label == "Stressé" else 0.0
        AUDIO_STRESS_LEVEL.set(stress_val)
        
        return {"emotion": result_label, "confidence": round(confidence, 4)}

    def train_custom_model(self, df, epochs=50, batch_size=16):
        """Entraînement MLOps avec calcul d'Accuracy et F1-Score sur jeu de test."""
        from sklearn.model_selection import train_test_split as sk_split
        print("🚀 Début de l'entraînement du modèle d'émotion (Architecture Binaire)...")
        self.classifier.train()

        X, y = [], []

        # 1. Préparation des données
        for _, row in df.iterrows():
            X.append([
                row['volume'],
                row['zcr'],
                row['spectral_centroid'],
                row['tempo'] / 200.0,
                row['pause_ratio']
            ])
            y.append(int(row['label']))

        # 2. Split train / test (80/20) — métriques calculées sur données jamais vues
        X_train, X_test, y_train, y_test = sk_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(self.device)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("Audionet_DL")

        with mlflow.start_run():
            # Boucle d'entraînement sur le jeu d'entraînement uniquement
            for epoch in range(epochs):
                total_loss = 0
                self.classifier.train()
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Évaluation sur le jeu de TEST (données jamais vues)
            self.classifier.eval()
            with torch.no_grad():
                outputs = self.classifier(X_test_t)
                _, predicted = torch.max(outputs.data, 1)
                all_preds  = predicted.cpu().numpy()
                all_labels = y_test_t.cpu().numpy()

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            # Sauvegarde et Logging
            model_path = "storage/models/emotion_net.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.classifier.state_dict(), model_path)
            
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_metric("final_loss", total_loss/len(dataloader))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            print(f"✅ Entraînement terminé. Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
            
            return {
                "status": "success", 
                "loss": round(total_loss/len(dataloader), 4),
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1, 4)
            }