import pandas as pd
import numpy as np
import json
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import os
import time
from src.monitoring.metrics import INFERENCE_TIME
from src.monitoring.mlflow.setup import init_mlflow
from src.monitoring.mlflow.utils import log_params, log_step_metrics, log_final_metrics, log_tags

class ScoringModel:
    def __init__(self, model_path="storage/models/scoring_rf.joblib"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Modèle Random Forest chargé.")
        else:
            print("Aucun modèle de scoring trouvé. Entraînement requis.")

    def train(self, df):
        """Entraîne la Random Forest pour prédire le score final (0-100)."""
        print("Entraînement du Random Forest Regressor...")

        # 1. Sélection des Features (Le "Vecteur de Fusion")
        # filler_rate (taux normalisé) remplace filler_count brut :
        # 7 tics sur 200 mots ≠ 7 tics sur 20 mots → la normalisation est indispensable.
        features = ['volume', 'tempo', 'pause_ratio', 'sentiment', 'filler_rate', 'stress_level']
        X = df[features]
        y = df['target_score']

        # 2. Split Train/Test pour avoir de VRAIES métriques
        # C'est crucial pour ne pas avoir 100% d'accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Configuration du modèle
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        # 4. Logging MLflow
        init_mlflow("Final_Scoring_ML")

        with mlflow.start_run():
            log_params({"n_estimators": 100, "max_depth": 10, "random_state": 42})
            log_tags({"model_type": "random_forest", "task": "score_regression"})

            # ── Courbe de progression par nombre d'arbres ─────────────────
            # Permet de voir dans MLflow quand la forêt commence à converger
            checkpoints = [1, 5, 10, 25, 50, 75, 100]
            for step, n in enumerate(checkpoints, start=1):
                rf_step = RandomForestRegressor(
                    n_estimators=n, max_depth=10, random_state=42, warm_start=False
                )
                rf_step.fit(X_train, y_train)
                preds_step = rf_step.predict(X_test)
                mae_step = mean_absolute_error(y_test, preds_step)
                r2_step  = r2_score(y_test, preds_step)
                log_step_metrics({
                    "mae":  round(float(mae_step), 4),
                    "r2":   round(float(r2_step),  4),
                }, step=step)
                print(f"  n_estimators={n:>3} — MAE: {mae_step:.3f}  R²: {r2_step:.3f}")

            # ── Modèle final (100 arbres) ─────────────────────────────────
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2  = r2_score(y_test, predictions)

            # ── K-Fold Cross-Validation (5 folds sur l'ensemble complet) ──
            # Donne une estimation plus fiable et moins bruitée que le simple split
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rf_cv = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            cv_mae_scores = -cross_val_score(rf_cv, X, y, cv=kf, scoring="neg_mean_absolute_error")
            cv_r2_scores = cross_val_score(rf_cv, X, y, cv=kf, scoring="r2")

            cv_mae_mean  = float(np.mean(cv_mae_scores))
            cv_mae_std   = float(np.std(cv_mae_scores))
            cv_r2_mean   = float(np.mean(cv_r2_scores))
            cv_r2_std    = float(np.std(cv_r2_scores))

            print(f"  Cross-Val (5-fold) — MAE: {cv_mae_mean:.3f} ± {cv_mae_std:.3f}  R²: {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")

            # Importance des features → loggée pour analyse dans MLflow
            feature_importance = dict(zip(features, self.model.feature_importances_))
            for fname, fval in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{fname}", round(float(fval), 4))

            log_final_metrics({
                "final_mae":      round(mae,         4),
                "final_r2":       round(r2,          4),
                "cv_mae_mean":    round(cv_mae_mean,  4),
                "cv_mae_std":     round(cv_mae_std,   4),
                "cv_r2_mean":     round(cv_r2_mean,   4),
                "cv_r2_std":      round(cv_r2_std,    4),
            })

            # Artefact JSON avec toutes les métriques d'évaluation
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_path = os.path.join(tmpdir, "evaluation_summary.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "holdout": {"mae": round(mae, 4), "r2": round(r2, 4)},
                        "cross_validation_5fold": {
                            "mae": {"mean": round(cv_mae_mean, 4), "std": round(cv_mae_std, 4)},
                            "r2":  {"mean": round(cv_r2_mean,  4), "std": round(cv_r2_std,  4)},
                        },
                        "feature_importance": {k: round(v, 4) for k, v in feature_importance.items()},
                    }, f, indent=2)
                mlflow.log_artifact(eval_path)

            # Sauvegarde
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            mlflow.log_artifact(self.model_path)

            print(f"Entraînement ML terminé. MAE: {mae:.2f}, R²: {r2:.2f}")

            return {
                "mae":         round(float(mae),        4),
                "r2":          round(float(r2),         4),
                "cv_mae_mean": round(float(cv_mae_mean), 4),
                "cv_r2_mean":  round(float(cv_r2_mean),  4),
            }

    def predict_score(self, audio_features, nlp_results, emotion_data):
        """Inférence : Fusionne les données en temps réel pour sortir la note."""
        if self.model is None:
            return 50.0  # Score par défaut si pas de modèle
            
        # Reconstruction du vecteur de fusion identique à l'entraînement
        # Note : emotion_data['emotion'] est converti en 1.0 (Stressé) ou 0.0 (Calme)
        stress_val = 1.0 if emotion_data.get('emotion') == "Stressé" else 0.0

        # filler_rate normalisé : critère principal du scoring (30% du poids)
        word_count = max(1, nlp_results.get('word_count', 1))
        filler_rate = nlp_results.get('filler_count', 0) / word_count

        input_data = pd.DataFrame([{
            'volume':       audio_features.get('volume', 0),
            'tempo':        audio_features.get('tempo', 0),
            'pause_ratio':  audio_features.get('pause_ratio', 0),
            'sentiment':    nlp_results.get('sentiment_score', 0),
            'filler_rate':  round(filler_rate, 4),
            'stress_level': stress_val,
        }])

        start_time = time.time()
        score = self.model.predict(input_data)[0]
        INFERENCE_TIME.labels(model_name='rf_scoring').observe(time.time() - start_time)
        return score