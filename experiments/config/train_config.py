from src import constants as const
from experiments.config  import global_config

default_params = {
    # "batch_size": 16,
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

default_V1_0_params = {
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
    },
    "selected_classes": {
        "asphalt-concrete": [
            const.EXCELLENT,
            const.GOOD,
            const.INTERMEDIATE,
            const.BAD,
        ],
        "paving_stones-sett": [
            const.EXCELLENT,
            const.GOOD,
            const.INTERMEDIATE,
            const.BAD,
        ],
        const.UNPAVED: [
            const.INTERMEDIATE,
            const.BAD,
            const.VERY_BAD,
        ],
    },
}

efficientnet_surface_params = {
    **global_config.global_config,
    **default_params,
    **default_V1_0_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "learning_rate": 0.0005,
    "dataset": "V1_0/downsampled_rtk",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split_downsampled_rtk.csv",
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    # "transform": {
    #     **global_config.global_config.get("transform"),
    #     "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    #     "crop": "lower_half_rtk",
    # },
    # "gpu_kernel": 0,
}

efficientnet_quality_params = {
    **global_config.global_config,
    **default_params,
    **default_V1_0_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "efficientnet",
    "level": const.SMOOTHNESS,
    "model": const.EFFNET_LINEAR,
    "learning_rate": 0.0005,
    "dataset": "V1_0/downsampled_rtk",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split_downsampled_rtk.csv",
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    # "transform": {
    #     **global_config.global_config.get("transform"),
    #     "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    #     "crop": "lower_half_rtk",
    # },
    # "gpu_kernel": 0,
}

train_valid_split_params = {
    **global_config.global_config,
    "dataset": "V1_0/downsampled_rtk", # TODO
    "metadata": "V1_0/metadata", # TODO
    "train_valid_split_list": "train_valid_split_downsampled_rtk.csv",
}

default_rtk_params = {
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
    },
    "selected_classes": {
        const.ASPHALT: [
            const.GOOD,
            const.REGULAR,
            const.BAD,
        ],
        const.PAVED: [
            const.GOOD,
            const.REGULAR,
            const.BAD,
        ],
        const.UNPAVED: [
            const.REGULAR,
            const.BAD,
        ],
    },
}

efficientnet_surface_params_rtk = {
    **global_config.global_config,
    **default_params,
    **default_rtk_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "rtk_efficientnet",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "learning_rate": 0.0005,
    "dataset": "RTK/GT",
    "metadata": "RTK/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    # "gpu_kernel": 0,
}

efficientnet_quality_params_rtk = {
    **global_config.global_config,
    **default_params,
    **default_rtk_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "rtk_efficientnet",
    "level": const.SMOOTHNESS,
    "model": const.EFFNET_LINEAR,
    "learning_rate": 0.0005,
    "dataset": "RTK/GT",
    "metadata": "RTK/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    # "gpu_kernel": 0,
}

train_valid_split_params_rtk = {
    **global_config.global_config,
    **default_rtk_params,
    "dataset": "RTK/GT", # TODO
    "metadata": "RTK/metadata", # TODO
    "train_valid_split_list": "train_valid_split.csv",
}

efficientnet_surface_params_V1_0all = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SURFACE_FIXED,
    "name": "efficientnet_a",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "learning_rate": 0.0005,
    "dataset": "V1_0/annotated",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
    },
    "gpu_kernel": 1,
}

efficientnet_quality_params_V1_0all = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "efficientnet_a",
    "level": const.SMOOTHNESS,
    "model": const.EFFNET_LINEAR,
    "learning_rate": 0.0005,
    "dataset": "V1_0/annotated",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    # "augment": {
    #     **global_config.global_config.get("augment"),
    #     "gaussian_blur_kernel": 11,
    #     "gaussian_blur_sigma": 5,
    # },
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
    },
    "gpu_kernel": 0,
}

train_valid_split_params_V1_0all = {
    **global_config.global_config,
    "dataset": "V1_0/annotated", # TODO
    "metadata": "V1_0/metadata", # TODO
    "train_valid_split_list": "train_valid_split.csv",
    "gpu_kernel": 0,
}