import os
import mlflow


def init_mlflow(experiment_name: str) -> None:
    """
    Configure le tracking URI MLflow et sélectionne l'expérience.
    À appeler en début de chaque run d'entraînement.

    Args:
        experiment_name: nom de l'expérience MLflow (é.g. 'Audionet_DL')
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
