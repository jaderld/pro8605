import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("InterviewFlowAI")
# Example usage: mlflow.log_metric, mlflow.log_param, etc.
