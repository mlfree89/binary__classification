import mlflow


def start_run(experiment_name):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_results(params, metrics):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
