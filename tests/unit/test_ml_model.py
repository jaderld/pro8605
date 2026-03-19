import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.models.ml_model import ScoringModel


class TestScoringModelDefault(unittest.TestCase):
    """Tests sans modèle entraîné (score par défaut)."""

    def setUp(self):
        self.model = object.__new__(ScoringModel)
        self.model.model = None
        self.model.model_path = "storage/models/scoring_rf.joblib"

    def test_predict_default_score(self):
        score = self.model.predict_score({}, {}, {})
        self.assertEqual(score, 50.0)

    def test_predict_default_score_with_calm(self):
        score = self.model.predict_score(
            {'volume': 0.06, 'tempo': 110.0, 'pause_ratio': 0.2},
            {'sentiment_score': 0.3, 'filler_count': 2},
            {'emotion': 'Calme'}
        )
        self.assertEqual(score, 50.0)


class TestScoringModelTrained(unittest.TestCase):
    """Tests avec entraînement sur données synthétiques (MLflow mocké)."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 120
        cls.df = pd.DataFrame({
            'volume': np.random.uniform(0.01, 0.15, n),
            'tempo': np.random.uniform(80, 160, n),
            'pause_ratio': np.random.uniform(0.05, 0.45, n),
            'sentiment': np.random.uniform(-1, 1, n),
            'filler_rate': np.random.uniform(0.0, 0.15, n),
            'stress_level': np.random.randint(0, 2, n),
            'target_score': np.random.uniform(20, 95, n),
        })
        cls.model = object.__new__(ScoringModel)
        cls.model.model = None
        cls.model.model_path = "storage/models/scoring_rf.joblib"

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=None)
        mock_run.__exit__ = MagicMock(return_value=False)
        with patch('src.models.ml_model.mlflow') as mock_mlflow, \
             patch('src.models.ml_model.joblib') as mock_joblib, \
             patch('src.models.ml_model.init_mlflow'), \
             patch('src.models.ml_model.log_params'), \
             patch('src.models.ml_model.log_step_metrics'), \
             patch('src.models.ml_model.log_final_metrics'), \
             patch('src.models.ml_model.log_tags'):
            mock_mlflow.start_run.return_value = mock_run
            cls.metrics = cls.model.train(cls.df)

    def test_train_returns_mae(self):
        self.assertIn('mae', self.metrics)
        self.assertGreaterEqual(self.metrics['mae'], 0)

    def test_train_returns_r2(self):
        self.assertIn('r2', self.metrics)

    def test_predict_score_in_range(self):
        score = self.model.predict_score(
            {'volume': 0.07, 'tempo': 115.0, 'pause_ratio': 0.18},
            {'sentiment_score': 0.4, 'filler_count': 1},
            {'emotion': 'Calme'}
        )
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_predict_stressed_different_from_calm(self):
        calm = self.model.predict_score(
            {'volume': 0.07, 'tempo': 110.0, 'pause_ratio': 0.15},
            {'sentiment_score': 0.5, 'filler_count': 0},
            {'emotion': 'Calme'}
        )
        stressed = self.model.predict_score(
            {'volume': 0.07, 'tempo': 110.0, 'pause_ratio': 0.15},
            {'sentiment_score': 0.5, 'filler_count': 0},
            {'emotion': 'Stressé'}
        )
        # Le modèle doit distinguer les deux émotions (scores différents)
        self.assertNotEqual(round(stressed, 2), round(calm, 2))


if __name__ == '__main__':
    unittest.main()
