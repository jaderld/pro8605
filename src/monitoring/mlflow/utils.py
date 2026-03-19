import mlflow


def log_params(params: dict) -> None:
    """Enregistre des hyperparamètres dans le run MLflow actif."""
    if params:
        mlflow.log_params(params)


def log_step_metrics(metrics: dict, step: int) -> None:
    """
    Enregistre des métriques pour une étape donnée (epoch, fold, etc.).
    Permet de tracer des courbes d'évolution dans l'UI MLflow.

    Args:
        metrics: dictionnaire {nom_metrique: valeur}
        step:    numéro de l'étape (epoch ou fold)
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_final_metrics(metrics: dict) -> None:
    """Enregistre les métriques finales sans numéro d'étape."""
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_tags(tags: dict) -> None:
    """Enregistre des tags sur le run MLflow actif."""
    if tags:
        mlflow.set_tags(tags)
