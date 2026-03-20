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
import json
import tempfile
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
from src.monitoring.metrics import TRANSCRIPTION_TIME, AUDIO_STRESS_LEVEL, INFERENCE_TIME, MODEL_CONFIDENCE, PROCESSING_TIME
from src.monitoring.mlflow.setup import init_mlflow
from src.monitoring.mlflow.utils import log_params, log_step_metrics, log_final_metrics, log_tags

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
                print("Modèle d'émotion personnalisé (2 classes) chargé.")
            except Exception as e:
                print(f"⚠️ Erreur : {e}. Architecture incompatible ou fichier corrompu.")
        else:
            print("Aucun modèle d'émotion trouvé, utilisation des poids par défaut.")

    def transcribe_audio(self, audio_path):
        """Transcription via Whisper avec Prompt Engineering pour les tics."""
        start_time = time.time()
        audio, sr = librosa.load(audio_path, sr=16000)

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
        PROCESSING_TIME.labels(module='transcription').observe(elapsed_s)

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
        
        stress_val = 1.0 if result_label == "Stressé" else 0.0
        AUDIO_STRESS_LEVEL.set(stress_val)
        
        return {"emotion": result_label, "confidence": round(confidence, 4)}

    def train_custom_model(self, df, epochs=50, batch_size=16):
        """Entraînement MLOps avec calcul d'Accuracy et F1-Score sur jeu de test."""
        from sklearn.model_selection import train_test_split as sk_split
        print("Début de l'entraînement du modèle d'émotion (Architecture Binaire)...")
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

        init_mlflow("Audionet_DL")

        with mlflow.start_run():
            log_params({"epochs": epochs, "batch_size": batch_size, "lr": 0.001, "architecture": "SimpleAudioNet"})
            log_tags({"model_type": "pytorch", "task": "emotion_classification"})

            # ── Boucle d'entraînement : métriques loggées à chaque epoch ──
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

                avg_loss = total_loss / len(dataloader)

                # Évaluation à chaque epoch sur le jeu de TEST
                self.classifier.eval()
                with torch.no_grad():
                    test_outputs = self.classifier(X_test_t)
                    _, predicted = torch.max(test_outputs.data, 1)
                    epoch_preds  = predicted.cpu().numpy()
                    epoch_labels = y_test_t.cpu().numpy()

                epoch_acc = accuracy_score(epoch_labels, epoch_preds)
                epoch_f1  = f1_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)

                # Log par epoch → courbe dans MLflow UI
                log_step_metrics({
                    "train_loss":     round(avg_loss,   4),
                    "test_accuracy":  round(epoch_acc,  4),
                    "test_f1":        round(epoch_f1,   4),
                }, step=epoch + 1)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}  acc: {epoch_acc:.3f}  f1: {epoch_f1:.3f}")

            # ── Métriques finales & évaluation complète ───────────────────
            final_precision = precision_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)
            final_recall    = recall_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)
            cm = confusion_matrix(epoch_labels, epoch_preds)
            report = classification_report(
                epoch_labels,
                epoch_preds,
                target_names=["Calme", "Stressé"],
                zero_division=0,
            )

            log_final_metrics({
                "final_loss":      round(float(avg_loss),       4),
                "final_accuracy":  round(float(epoch_acc),      4),
                "final_f1":        round(float(epoch_f1),       4),
                "final_precision": round(float(final_precision), 4),
                "final_recall":    round(float(final_recall),    4),
                # Métriques par classe : TN, FP, FN, TP (matrice 2×2)
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            })

            # Sauvegarde des artefacts d'évaluation dans MLflow
            with tempfile.TemporaryDirectory() as tmpdir:
                # Rapport texte
                report_path = os.path.join(tmpdir, "classification_report.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)

                # Matrice de confusion JSON
                cm_path = os.path.join(tmpdir, "confusion_matrix.json")
                with open(cm_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "labels": ["Calme", "Stressé"],
                        "matrix": cm.tolist(),
                    }, f, indent=2)
                mlflow.log_artifact(cm_path)

            # Sauvegarde du modèle
            model_path = "storage/models/emotion_net.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.classifier.state_dict(), model_path)
            mlflow.log_artifact(model_path)

            print(f"Entraînement terminé. Accuracy: {epoch_acc:.2f}  F1: {epoch_f1:.2f}  Précision: {final_precision:.2f}")
            print(f"   Matrice de confusion :\n{cm}")

            return {
                "status":    "success",
                "loss":      round(float(avg_loss),        4),
                "accuracy":  round(float(epoch_acc),       4),
                "f1_score":  round(float(epoch_f1),        4),
                "precision": round(float(final_precision), 4),
                "recall":    round(float(final_recall),    4),
            }