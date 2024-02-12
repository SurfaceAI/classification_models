from src import constants as const

sweep_method = {"method": "bayes"}

sweep_metric_loss = {"metric": {"name": "eval/loss", "goal": "minimize"}}

sweep_metric_acc = {"metric": {"name": "eval/acc", "goal": "maximize"}}

sweep_params = {
    "batch_size": {"values": [16, 48, 128]},
    "epochs": {"value": 20},
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},
    "optimizer_cls": {"value": const.OPTI_ADAM},
}

fixed_params = {
    "batch_size": 16,  # 48
    "epochs": 10,
    "learning_rate": 0.0001,
    "optimizer_cls": const.OPTI_ADAM,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
}
