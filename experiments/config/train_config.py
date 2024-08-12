from src import constants as const
from experiments.config  import global_config

default_params = {
    "batch_size": 64, #16,  # 48
    "epochs": 2,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "is_regression": False,
    "is_hierarchical": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "max_class_size": None,
    "lr_scheduler": True,
    "freeze_convs": True,
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
    "batch_size": 16,
    "epochs": 1,
    "learning_rate": 0.003,
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

vgg16_surface_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "vgg16_coarse",
    "level": const.SURFACE,
    "model": const.VGG16,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "learning_rate": 0.00056,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    # "gpu_kernel": 0,
}

vgg16_quality_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "vgg16_fine",
    "level": const.SMOOTHNESS,
    "model": const.VGG16,
    "learning_rate": 0.0006,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_MSE,
    # "gpu_kernel": 0,
}

efficientnet_surface_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
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
    "project": const.PROJECT_MULTI_LABEL_FIXED,
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

# efficientnet_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_SURFACE_FIXED,
#     "name": "efficientnet",
#     "level": const.SURFACE,
#     "model": const.EFFICIENTNET,
#     # "max_class_size": 2,
# }

# efficientnet_flatten_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_FLATTEN_FIXED,
#     "name": "efficientnet",
#     "level": const.FLATTEN,
#     "model": const.EFFICIENTNET,
# }

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
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 0.00003,
    "is_regression": True,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "clm": False,
    "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
    "name": "VGG16_regression_asphalt",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V12/annotated/asphalt",
    "model": const.VGG16,

}

vgg16_asphalt_classification_params = {
    **global_config.global_config,
    **default_params,
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 0.00003,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "clm": False,
    "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
    "name": "VGG16_classification_asphalt",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V12/annotated/asphalt",
    "model": const.VGG16,

}

vgg16_asphalt_CLM_params = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.01,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "clm": True,
    "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
    "name": "VGG16_CLM_asphalt",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V12/annotated/asphalt",
    "model": const.VGG16_CLM,
}

vgg16_asphalt_Rosati_params = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.0001,
    "is_regression": False,
    "eval_metric": const.EVAL_METRIC_ACCURACY,
    "clm": True,
    "two_optimizers":False,
    "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
    "name": "VGG16_CLM_asphalt",
    "level": const.ASPHALT,
    "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
    "dataset": "V12/annotated/asphalt",
    "model": const.VGG16_CLM,
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
    "dataset": "V12/annotated/asphalt",
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
    "dataset": "V12/annotated/asphalt",
    "model": const.EFFNET_LINEAR,

}

B_CNN = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "B_CNN",
    "level": const.HIERARCHICAL,
    "model": const.BCNN,
}

B_CNN_PRE = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "B_CNN_pretrained",
    "level": const.HIERARCHICAL,
    "model": const.BCNN_PRE,
}

C_CNN = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "Condition_CNN",
    "level": const.HIERARCHICAL,
    "model": const.CCNN,
}

C_CNN_PRE = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "Condition_CNN_pretrained",
    "level": const.HIERARCHICAL,
    "model": const.CCNN_PRE,
}

H_NET = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "HiearchyNet",
    "level": const.HIERARCHICAL,
    "model": const.HNET,
}

H_NET_PRE = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "HierarchyNet_regression",
    "level": const.HIERARCHICAL,
    "model": const.HNET_PRE,
    "head": 'regression', #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": None, #'use_ground_truth', 'use_condition_layer', 'top_coarse_prob'
}


GH_CNN = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "GH_CNN",
    "level": const.HIERARCHICAL,
    "model": const.GHCNN,
}

GH_CNN_PRE = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "GH_CNN_classification",
    "level": const.HIERARCHICAL,
    "model": const.GHCNN_PRE,
    "head": 'classification', #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": None, #'use_ground_truth', 'use_condition_layer', 'top_coarse_prob'
}


B_CNN_regression = {
    **global_config.global_config,
    **default_params,
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.001,
    "optimizer": const.OPTI_ADAM,
    "is_regression": True,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "B_CNN_Regression",
    "level": const.HIERARCHICAL,
    "model": const.BCNNREGRESSION,
    "hierarchy_method": const.WEIGHTEDSUM
}


C_CNN_CLM = {
    **global_config.global_config,
    **default_params,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "C_CNN_classification",
    "level": const.HIERARCHICAL,
    "model": const.CCNNCLMPRE,
    "head": 'classification', #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": 'use_model_structure', #'use_ground_truth', 'None'
}

B_CNN_CLM = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "B_CNN_CLM",
    "level": const.HIERARCHICAL,
    "model": const.BCNN_PRE,
    "head": 'clm', #'regression', 'classification', 'corn', 'clm'
    "hierarchy_method": 'b_cnn', #'use_ground_truth', 'use_condition_layer', 'b_cnn'
    "lw_modifier": True,
}

