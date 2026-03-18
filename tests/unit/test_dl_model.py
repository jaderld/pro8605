import unittest
import numpy as np
import torch
from src.models.dl_model import SimpleAudioNet, InterviewModel


class TestSimpleAudioNet(unittest.TestCase):
    """Tests du réseau de neurones SimpleAudioNet (sans Whisper)."""

    def setUp(self):
        self.model = SimpleAudioNet(input_dim=5, num_classes=2)
        self.model.eval()

    def test_forward_output_shape(self):
        x = torch.rand(10, 5)
        out = self.model(x)
        self.assertEqual(out.shape, (10, 2))

    def test_single_sample_forward(self):
        x = torch.rand(1, 5)
        out = self.model(x)
        self.assertEqual(out.shape, (1, 2))


class TestPredictEmotion(unittest.TestCase):
    """Tests de predict_emotion en isolant InterviewModel du chargement Whisper."""

    def setUp(self):
        # Instanciation sans __init__ pour éviter le chargement de Whisper
        self.interview_model = object.__new__(InterviewModel)
        self.interview_model.device = "cpu"
        self.interview_model.classifier = SimpleAudioNet(5, 2)
        self.interview_model.classifier.eval()

    def test_predict_emotion_returns_keys(self):
        features = np.array([0.05, 0.01, 0.5, 0.6, 0.2])
        result = self.interview_model.predict_emotion(features)
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)

    def test_predict_emotion_valid_label(self):
        features = np.array([0.05, 0.01, 0.5, 0.6, 0.2])
        result = self.interview_model.predict_emotion(features)
        self.assertIn(result['emotion'], ['Calme', 'Stressé'])

    def test_predict_emotion_confidence_range(self):
        features = np.array([0.05, 0.01, 0.5, 0.6, 0.2])
        result = self.interview_model.predict_emotion(features)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
