from src import constants as const
from experiments.config  import global_config

default_params = {
    # "batch_size": 16,  # 48
    "valid_batch_size": 48,
    "epochs": 20,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "save_state": True,
    # "is_regression": False,
    # "eval_metric": const.EVAL_METRIC_ACCURACY,
}

default_search_params = {
    "batch_size": {"values": [16, 48, 128]},
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 1e-03},
}

efficientnet_params_example = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
}

efficientnet_sweep_params_example = {
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

efficientnet_surface_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "learning_rate": 0.00056,
    "dataset": "V1_0/train",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    # "gpu_kernel": 0,
}

efficientnet_quality_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "efficientnet",
    "level": const.SMOOTHNESS,
    "model": const.EFFNET_LINEAR,
    "learning_rate": 0.0006,
    "dataset": "V1_0/train",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    # "gpu_kernel": 0,
}

train_valid_split_params = {
    **global_config.global_config,
    "dataset": "V1_0/train", # TODO
    "metadata": "V1_0/metadata", # TODO
    "train_valid_split_list": "train_valid_split.csv",
}