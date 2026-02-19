import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

class ScoringModel:
    def __init__(self, model_path='storage/models/scoring_rf.pkl'):
        self.model_path = model_path
        # On remplace la LogisticRegression par un RandomForestRegressor pour avoir une note sur 100
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self._load_model()

    def _load_model(self):
        """Charge le modèle s'il existe déjà sur le disque."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Modèle ML chargé avec succès.")
        else:
            print("⚠️ Aucun modèle ML trouvé. L'entraînement est nécessaire.")

    def train(self, df: pd.DataFrame):
        print("🚀 Début de l'entraînement du modèle ML...")
        
        features = ['filler_count', 'pause_ratio', 'sentiment']
        target = 'target_score'
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Configurer l'URI de tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("Interview_Scoring")

        with mlflow.start_run():
            # Entraînement
            self.model.fit(X_train, y_train)

            # Évaluation
            predictions = self.model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
            # --- CORRECTION ICI ---
            # On log les paramètres et métriques individuellement
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_metric("r2_score", r2)
            
            # Au lieu de log_model qui peut échouer en 404, on log juste l'artefact 
            # ou on utilise une version simplifiée :
            try:
                mlflow.sklearn.log_model(self.model, "model")
            except Exception as e:
                print(f"⚠️ Warning: Impossible de loguer le modèle complet sur MLflow: {e}")
            # -----------------------

            print(f"✅ Entraînement terminé. R2 Score : {r2:.2f}")

        # 3. Sauvegarde physique du modèle (comme ton ancien save)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"💾 Modèle sauvegardé dans {self.model_path}")

    def predict_score(self, audio_features: dict, nlp_results: dict) -> float:
        """
        Méthode utilisée par l'API pour noter un nouveau candidat en temps réel.
        """
        # Sécurité au cas où l'API est appelée avant l'entraînement
        if not hasattr(self.model, 'estimators_'):
            return 50.0 

        # Formatage des données reçues de l'API
        input_data = pd.DataFrame([{
            'filler_count': nlp_results.get('filler_count', 0),
            'pause_ratio': audio_features.get('pause_ratio', 0.1),
            'sentiment': nlp_results.get('sentiment_score', 0.0)
        }])

        # Prédiction de la note
        score = self.model.predict(input_data)[0]
        return round(float(score), 2)