import mlflow

def log_experiment(metrics: dict, params: dict = None, tags: dict = None):
    with mlflow.start_run():
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)
