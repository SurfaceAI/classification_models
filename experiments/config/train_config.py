from src import constants as const
from experiments.config  import global_config

default_params = {
    "batch_size": 16,  # 48
    "epochs": 10,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
}

default_search_params = {
    "batch_size": {"values": [16, 48, 128]},
    "epochs": {"value": 20},
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},
    "optimizer": {"value": const.OPTI_ADAM},
}


vgg16_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "VGG16",
    "level": const.SURFACE,
    "model": const.VGG16,
}

vgg16_smoothness_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "VGG16",
    "level": const.SMOOTHNESS,
    "model": const.VGG16,
}

vgg16_flatten_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_FLATTEN_FIXED,
    "name": "VGG16",
    "level": const.FLATTEN,
    "model": const.VGG16,
}

rateke_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "rateke",
    "level": const.SURFACE,
    "model": const.RATEKE,
}

rateke_flatten_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_FLATTEN_FIXED,
    "name": "rateke",
    "level": const.FLATTEN,
    "model": const.RATEKE,
}

efficientnet_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFICIENTNET,
}

efficientnet_flatten_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_FLATTEN_FIXED,
    "name": "efficientnet",
    "level": const.FLATTEN,
    "model": const.EFFICIENTNET,
}

sweep_params = {
    **global_config.global_config,
    **default_params,
    "method": "bayes",
    "metric": {"name": "eval/acc", "goal": "maximize"},
    "search_params": {**default_search_params,
                      "model": {"values": [const.VGG16, const.EFFICIENTNET]},                 
                     },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "VGG16-efficientnet",
    "level": const.SURFACE,
    "sweep_counts": 10,
}

vgg16_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.VGG16,
    "method": "bayes",
    "metric": {"name": "eval/acc", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "VGG16",
    "level": const.SURFACE,
    "sweep_counts": 10,
}

efficientnet_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.EFFICIENTNET,
    "method": "bayes",
    "metric": {"name": "eval/acc", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "efficientnet",
    "level": const.SURFACE,
    "sweep_counts": 10,
}

rateke_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.RATEKE,
    "method": "bayes",
    "metric": {"name": "eval/acc", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "rateke",
    "level": const.SURFACE,
    "sweep_counts": 10,
}

vgg16_regression_params = {
    **global_config.global_config,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "VGG16_Regression",
    "level": const.SMOOTHNESS,
    "model": const.VGG16,

}
