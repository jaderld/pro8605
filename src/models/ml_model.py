import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import os

class ScoringModel:
    def __init__(self, model_path="storage/models/scoring_rf.joblib"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("✅ Modèle Random Forest chargé.")
        else:
            print("ℹ️ Aucun modèle de scoring trouvé. Entraînement requis.")

    def train(self, df):
        """Entraîne la Random Forest pour prédire le score final (0-100)."""
        print("🚀 Entraînement du Random Forest Regressor...")

        # 1. Sélection des Features (Le "Vecteur de Fusion")
        # On utilise tout : Audio + NLP + Emotion (stress_level)
        features = ['volume', 'tempo', 'pause_ratio', 'sentiment', 'filler_count', 'stress_level']
        X = df[features]
        y = df['target_score']

        # 2. Split Train/Test pour avoir de VRAIES métriques
        # C'est crucial pour ne pas avoir 100% d'accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Configuration du modèle
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        # 4. Logging MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("Final_Scoring_ML")

        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            
            # Prédictions sur le jeu de test
            predictions = self.model.predict(X_test)
            
            # Calcul des métriques
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Sauvegarde du modèle
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

            # Log dans MLflow
            mlflow.log_params({"n_estimators": 100, "max_depth": 10})
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            print(f"✅ Entraînement ML terminé. MAE: {mae:.2f}, R2: {r2:.2f}")

            # --- C'EST CETTE LIGNE QUI MANQUAIT À TON API ---
            return {
                "mae": round(float(mae), 4),
                "r2": round(float(r2), 4)
            }

    def predict_score(self, audio_features, nlp_results, emotion_data):
        """Inférence : Fusionne les données en temps réel pour sortir la note."""
        if self.model is None:
            return 50.0 # Score par défaut si pas de modèle
            
        # Reconstruction du vecteur de fusion identique à l'entraînement
        # Note : emotion_data['emotion'] est converti en 1.0 (Stressé) ou 0.0 (Calme)
        stress_val = 1.0 if emotion_data.get('emotion') == "Stressé" else 0.0
        
        input_data = pd.DataFrame([{
            'volume': audio_features.get('volume', 0),
            'tempo': audio_features.get('tempo', 0),
            'pause_ratio': audio_features.get('pause_ratio', 0),
            'sentiment': nlp_results.get('sentiment_score', 0),
            'filler_count': nlp_results.get('filler_count', 0),
            'stress_level': stress_val
        }])

        return self.model.predict(input_data)[0]