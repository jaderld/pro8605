import unittest
import numpy as np
from src.models.ml_model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.model = MLModel()
        self.X = np.random.rand(20, 3)
        self.y = np.random.randint(0, 2, 20)
        self.model.train(self.X, self.y)

    def test_predict(self):
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_evaluate(self):
        acc = self.model.evaluate(self.X, self.y)
        self.assertTrue(0 <= acc <= 1)

if __name__ == '__main__':
    unittest.main()
