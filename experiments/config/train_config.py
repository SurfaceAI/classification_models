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
        "normalize": (const.V11_ANNOTATED_MEAN,const.V11_ANNOTATED_SD)},
    "batch_size": 16,
    "is_regression": True,
    # "eval_metric": const.EVAL_METRIC_MSE,
    "project": const.PROJECT_SMOOTHNESS_FIXED,
    "name": "all_train_optim_lf_effnet_Reg",
    "model": const.EFFNET_LINEAR,
    "gpu_kernel": 0,
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
    "dataset": "V12/annotated",
    "metadata": "V12/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "level": const.SURFACE,
    "model": const.EFFNET_LINEAR,
    "gpu_kernel": 0,

}

effnet_surface_sweep_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 1,
    "epochs": 20,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    'model': const.EFFNET_LINEAR,
    "dataset": "V1_0/annotated",
    "method": "bayes",
    "metric": {"name": f"eval/{const.EVAL_METRIC_ACCURACY}", "goal": "maximize"},
    "search_params":
        {"batch_size": {"values": [16]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-04, "max": 0.001},},
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "effnet",
    "level": const.SURFACE,
    "sweep_counts": 2,
    "transform":
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,    
        "normalize": const.NORM_DATA,},    
    "save_state": True,
 }

effnet_surface_blur_sweep_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 1,
    "epochs": 30,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    'model': const.EFFNET_LINEAR,
    "dataset": "V1_0/annotated",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "method": "grid",
    "metric": {"name": f"eval/{const.EVAL_METRIC_ACCURACY}", "goal": "maximize"},
    "search_params":{
        "batch_size": {"values": [16]},
        # "learning_rate": {"values": [0.0003, 0.0007]},
        "learning_rate": {"values": [0.0003]},
        "augment": {
            "parameters": {
                **{key: {"value": value} for key, value in global_config.global_config.get("augment").items()},
                "gaussian_blur_kernel": {"values": [5, 7, 9, 11]},
                "gaussian_blur_sigma": {"values": [2, 3.5, 5]},
            }
        },
        },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "effnet_blur",
    "level": const.SURFACE,
    "sweep_counts": 100,
    "transform":
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,    
        "normalize": const.NORM_DATA,},    
    "save_state": True,
 }

effnet_surface_high_blur_sweep_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 0,
    "epochs": 30,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    'model': const.EFFNET_LINEAR,
    "dataset": "V1_0/annotated",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
    "method": "grid",
    "metric": {"name": f"eval/{const.EVAL_METRIC_ACCURACY}", "goal": "maximize"},
    "search_params":{
        "unused": {"values": [1, 2]},
        "batch_size": {"values": [16]},
        # "learning_rate": {"values": [0.0003, 0.0007]},
        "learning_rate": {"values": [0.0003, 0.0001653, 0.0006]}, # 0.0001653 used for first high blur experiment
        # "learning_rate": {"values": [0.0005284]}, # 0.0001653 used for first high blur experiment
        # "augment": {
        #     "parameters": {
        #         **{key: {"value": value} for key, value in global_config.global_config.get("augment").items()},
        #         "gaussian_blur_kernel": {"values": [5]},
        #         "gaussian_blur_sigma": {"values": [2]},
        #     }
        # },
        },
    "project": const.PROJECT_SURFACE_SWEEP,
    "name": "effnet_blur",
    "level": const.SURFACE,
    "sweep_counts": 100,
    "transform":
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,    
        "normalize": const.NORM_DATA,},    
    "save_state": True,
    # "checkpoint_top_n": 10,
 }

effnet_quality_sweep_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 1,
    "epochs": 20,
    "eval_metric": const.EVAL_METRIC_MSE,
    'model': const.EFFNET_LINEAR,
    "dataset": "V1_0/annotated",
    "method": "bayes",
    "is_regression": True,
    "metric": {"name": f"eval/{const.EVAL_METRIC_MSE}", "goal": "minimize"},
    "search_params":
        {"batch_size": {"values": [16]},
        # "learning_rate": {"distribution": "log_uniform_values", "min": 1e-04, "max": 0.001},},
        "learning_rate": {"values": [0.0006]},},
    "project": const.PROJECT_SMOOTHNESS_SWEEP,
    "name": "effnet_reg",
    "level": const.SMOOTHNESS,
    "sweep_counts": 1,
    "transform": 
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": const.NORM_DATA,},  
    "save_state": True,
}

road_scenery_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 1,
    "epochs": 20,
    "batch_size": 16,  # 48
    "learning_rate": 0.0003,
    'model': const.EFFNET_LINEAR,
    "dataset": "road_scenery",
    "project": const.PROJECT_SCENERY_FIXED,
    "name": "effnet_scenery",
    "level": const.FLATTEN,
    "transform":
        {"resize": (384, 384),
        "crop": const.CROP_LOWER_HALF,    
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),},   
    "selected_classes": {
        '1_1_road': [
            '1_1_parking_area',
            '1_1_rails_on_road',
            '1_1_road_general',
        ],
        '1_2_cycleway': [
            '1_2_hochbord',
            '1_2_lane',
        ],
        '1_3_pedestrian': [
            '1_3_pedestrian_area',
            '1_3_railway_platform',
            '1_3_sidewalk',
        ],
        '1_4_path': [
            '1_4_path_unspecified',
            '1_4_trampling_trail',
        ],
        '2_1_no_focus': [
            '2_1_other',
            '2_1_vertical',
        ],
        '2_2_no_street': [
            '2_2_all'
        ],
    },
    "save_state": True,
 }

road_scenery_focus_params = {
    **global_config.global_config,
    **default_params,
    'gpu_kernel': 1,
    "epochs": 20,
    "batch_size": 16,  # 48
    "learning_rate": 0.0003,
    'model': const.EFFNET_LINEAR,
    "dataset": "road_scenery",
    "project": const.PROJECT_SCENERY_FIXED,
    "name": "effnet_scenery_focus",
    "level": const.FLATTEN,
    "transform":
        {"resize": (384, 384),
        # "crop": const.CROP_LOWER_HALF,    
        "crop": None,    
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),},   
    "selected_classes": {
        '1_1_road': [
            '1_1_rails_on_road',
            '1_1_road_general',
        ],
        '1_2_bicycle': [
            '1_2_cycleway',
            '1_2_lane',
        ],
        '1_3_pedestrian': [
            '1_3_pedestrian_area',
            '1_3_railway_platform',
            '1_3_footway',
        ],
        '1_4_path': [
            '1_4_path_unspecified',
        ],
        '2_1_no_focus_no_street': [
            '2_1_all'
        ],
    },
    "save_state": True,
    "avg_pool": 4,
 }

train_valid_split_params = {
    **global_config.global_config,
    # "root_data": str(global_config.ROOT_DIR / "data" / "training"),
    "dataset": "V1_0/annotated",
    "metadata": "V1_0/metadata",
    "train_valid_split_list": "train_valid_split.csv",
}