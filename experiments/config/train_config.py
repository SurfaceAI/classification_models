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
    "batch_size": 96,
    "epochs": 20,
    "learning_rate": 0.00003,
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
    # "max_class_size": 2,
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
    "sweep_counts": 30,
    "save_state": False,
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

vgg16_asphalt_regression_params = {
    **global_config.global_config,
    **default_params,
    "batch_size": 96,
    "epochs": 20,
    "learning_rate": 0.00003,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "VGG16_Regression",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V7/annotated/asphalt",
    "model": const.VGG16,

}

vgg16_asphalt_crophalf_regression_params = {
    **global_config.global_config,
    **default_params,
    "transform": 
        {"resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": const.NORM_DATA,},
    "batch_size": 96,
    "epochs": 20,
    "learning_rate": 0.00003,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "VGG16_Regression",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V7/annotated/asphalt",
    "model": const.VGG16,

}

effnet_asphalt_crophalf_regression_params = {
    **global_config.global_config,
    **default_params,
    "transform": 
        {"resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": const.NORM_DATA,},
    "batch_size": 96,
    "epochs": 20,
    "learning_rate": 0.0001,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "effnet_Regression",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V7/annotated/asphalt",
    "model": const.EFFNET_LINEAR,

}

effnet_quality_regression_params = {
    **global_config.global_config,
    **default_params,
    "transform": 
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V9_ANNOTATED_MEAN,const.V9_ANNOTATED_SD)},
    "batch_size": 16,
    "is_regression": True,
    # "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "all_train_optim_lf_effnet_Reg",
    "model": const.EFFNET_LINEAR,

}

effnet_surface_params = {
    **global_config.global_config,
    **default_params,
    # "gpu_kernel": 0,
    "transform": 
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": const.NORM_DATA,},
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 0.00037,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "all_train_optim_lr_effnet_linear",
    "dataset": "V9/annotated",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "gpu_kernel": 0,

}

effnet_surface_sweep_params = {
    **global_config.global_config,
    **default_params,
    # 'gpu_kernel': 0,
    "epochs": 20,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    'model': const.EFFNET_LINEAR,
    "dataset": "V9/annotated",
    "method": "bayes",
    "metric": {"name": f"eval/{const.EVAL_METRIC_ACCURACY}", "goal": "maximize"},
    "search_params":
        {"batch_size": {"values": [16]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},},
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "optim_lf_effnet",
    "level": const.SURFACE,
    "sweep_counts": 5,
    "transform":
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,    
        "normalize": const.NORM_DATA,},  
 }

effnet_quality_sweep_params = {
    **global_config.global_config,
    **default_params,
    # 'gpu_kernel': 0,
    "epochs": 20,
    "eval_metric": const.EVAL_METRIC_MSE,
    'model': const.EFFNET_LINEAR,
    "dataset": "V9/annotated",
    "method": "bayes",
    "is_regression": True,
    "metric": {"name": f"eval/{const.EVAL_METRIC_MSE}", "goal": "minimize"},
    "search_params":
        {"batch_size": {"values": [16]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},},
    "project": const.PROJECT_SMOOTHNESS_SWEEP,
    "name": "optim_lf_effnet_reg",
    "level": const.SMOOTHNESS,
    "sweep_counts": 5,
    "transform": 
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": const.NORM_DATA,},  
}
