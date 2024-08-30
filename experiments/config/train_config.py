from src import constants as const
from experiments.config  import global_config

default_params = {
    "batch_size": 16,
    "valid_batch_size": 48,
    "epochs": 20,
    "learning_rate": 0.0005,
    "optimizer": const.OPTI_ADAM,
}

default_search_params = {
    "batch_size": {"values": [16, 48, 128]},
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 1e-03},
}

params_example = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
}

sweep_params_example = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "method": "bayes",
    "metric": {"name": f"eval/{const.EVAL_METRIC_ACCURACY}", "goal": "maximize"},
    "search_params": {
        **default_search_params,                 
                     },
    "sweep_counts": 10,
}

type_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "dataset": "V1_0/train",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
}

quality_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "efficientnet",
    "level": const.SMOOTHNESS,
    "model": const.EFFNET_LINEAR,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    "dataset": "V1_0/train",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
}

train_valid_split_params = {
    **global_config.global_config,
    "dataset": "V1_0/train",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
}
