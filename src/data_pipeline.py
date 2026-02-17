import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def extract(self):
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            return df
        else:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def transform(self, df):
        # Example: fillna, encode, scale
        df = df.fillna(0)
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        df[features] = self.scaler.fit_transform(df[features])
        return df

    def split(self, df, target, test_size=0.2, random_state=42):
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
