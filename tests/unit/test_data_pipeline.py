import unittest
import pandas as pd
from src.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DataPipeline()
        self.df = pd.DataFrame({
            'a': [1, 2, 3, None],
            'b': [4, 5, 6, 7],
            'target': [0, 1, 0, 1]
        })

    def test_transform(self):
        df_t = self.pipeline.transform(self.df)
        self.assertFalse(df_t.isnull().values.any())

    def test_split(self):
        X_train, X_test, y_train, y_test = self.pipeline.split(self.df, 'target')
        self.assertEqual(len(X_train) + len(X_test), len(self.df))

if __name__ == '__main__':
    unittest.main()
