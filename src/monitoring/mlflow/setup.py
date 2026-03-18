import os
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Audionet_DL")
# Example usage: mlflow.log_metric, mlflow.log_param, etc.
