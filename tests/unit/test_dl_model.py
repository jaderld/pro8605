import unittest
import numpy as np
from src.models.dl_model import DLModel

class TestDLModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.num_classes = 2
        self.model = DLModel(self.input_dim, self.num_classes)
        self.X = np.random.rand(10, self.input_dim)
        self.y = np.random.randint(0, self.num_classes, 10)
        self.model.train(self.X, self.y, epochs=1)

    def test_predict(self):
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

if __name__ == '__main__':
    unittest.main()
