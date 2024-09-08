from src import constants as const
from experiments.config  import global_config

default_params = {
    "batch_size": 64, #16,  # 48
    "epochs": 2,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "is_regression": False,
    "is_hierarchical": False,
    "eval_metric": const.EVAL_METRIC_ALL,
    #"max_class_size": None,
    "lr_scheduler": True,
    "freeze_convs": True,
}

default_search_params = {
    "batch_size": {"values": [16, 32, 64]},
    "epochs": {"value": 15},
    "learning_rate": {"values": [1e-05, 1e-04, 1e-03, 1e-02]},
    #"learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},
    "optimizer": {"value": const.OPTI_ADAM},
    "fc_neurons": {"values": [512, 1024, 2048]},
    "freeze_convs": {"values": [True, False]}
}


# vgg16_params = {
#     **global_config.global_config,
#     **default_params,
#     "batch_size": 16,
#     "epochs": 1,
#     "learning_rate": 0.003,
#     "project": const.PROJECT_SURFACE_FIXED,
#     "name": "VGG16",
#     "level": const.SURFACE,
#     "model": const.VGG16,
# }

# vgg16_smoothness_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_SMOOTHNESS_FIXED,
#     "name": "VGG16",
#     "level": const.SMOOTHNESS,
#     "model": const.VGG16,
# }

# vgg16_flatten_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_FLATTEN_FIXED,
#     "name": "VGG16",
#     "level": const.FLATTEN,
#     "model": const.VGG16,
# # }

# rateke_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_SURFACE_FIXED,
#     "name": "rateke",
#     "level": const.SURFACE,
#     "model": const.RATEKE,
# }

# rateke_flatten_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_FLATTEN_FIXED,
#     "name": "rateke",
#     "level": const.FLATTEN,
#     "model": const.RATEKE,
# }

vgg16_surface_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "CC_coarse",
    "level": const.SURFACE,
    "model": const.VGG16,
    "head": const.CLASSIFICATION,
    "eval_metric": const.EVAL_METRIC_ALL,
    "learning_rate": 0.00056,
    "hierarchy_method": const.CC,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    # "gpu_kernel": 0,
}

vgg16_quality_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "CC_fine_clm",
    "level": const.SMOOTHNESS,
    "model": const.VGG16,
    "learning_rate": 0.01,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    "head": const.CLM,
    "hierarchy_method": const.CC,
    "eval_metric": const.EVAL_METRIC_ALL,
    # "gpu_kernel": 0,
}

# efficientnet_surface_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_MULTI_LABEL_FIXED,
#     "name": "efficientnet",
#     "level": const.SURFACE,
#     "model": const.EFFNET_LINEAR,
#     "is_regression": False,
#     "eval_metric": const.EVAL_METRIC_ACCURACY,
#     "learning_rate": 0.00056,
#     "dataset": "V1_0/train",
#     "metadata": "V1_0/metadata",
#     "train_valid_split_list": "train_valid_split.csv",
#     # "gpu_kernel": 0,
# }

# efficientnet_quality_params = {
#     **global_config.global_config,
#     **default_params,
#     "project": const.PROJECT_MULTI_LABEL_FIXED,
#     "name": "efficientnet",
#     "level": const.SMOOTHNESS,
#     "model": const.EFFNET_LINEAR,
#     "learning_rate": 0.0006,
#     "dataset": "V1_0/train",
#     "metadata": "V1_0/metadata",
#     "train_valid_split_list": "train_valid_split.csv",
#     "is_regression": True,
#     "eval_metric": const.EVAL_METRIC_MSE,
#     # "gpu_kernel": 0,
# }

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

# sweep_params = {
#     **global_config.global_config,
#     **default_params,
#     "method": "bayes",
#     "metric": {"name": "eval/acc", "goal": "maximize"},
#     "search_params": {**default_search_params,
#                       "model": {"values": [const.VGG16, const.EFFICIENTNET]},                 
#                      },
#     "project": const.PROJECT_SURFACE_SWEEP,
#     "name": "VGG16-efficientnet",
#     "level": const.SURFACE,
#     "sweep_counts": 30,
#     "save_state": False,
# }

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

# efficientnet_sweep_params = {
#     **global_config.global_config,
#     **default_params,
#     'model': const.EFFICIENTNET,
#     "method": "bayes",
#     "metric": {"name": "eval/acc", "goal": "maximize"},
#     "search_params": {**default_search_params,                 
#                      },
#     "project": const.PROJECT_SURFACE_SWEEP,
#     "name": "efficientnet",
#     "level": const.SURFACE,
#     "sweep_counts": 10,
# }

# rateke_sweep_params = {
#     **global_config.global_config,
#     **default_params,
#     'model': const.RATEKE,
#     "method": "bayes",
#     "metric": {"name": "eval/acc", "goal": "maximize"},
#     "search_params": {**default_search_params,                 
#                      },
#     "project": const.PROJECT_SURFACE_SWEEP,
#     "name": "rateke",
#     "level": const.SURFACE,
#     "sweep_counts": 10,
# }

# vgg16_regression_params = {
#     **global_config.global_config,
#     "batch_size": 16,
#     "epochs": 10,
#     "learning_rate": 0.0001,
#     "optimizer": const.OPTI_ADAM,
#     "is_regression": True,
#     "eval_metric": const.EVAL_METRIC_MSE,
#     "project": const.PROJECT_SMOOTHNESS_FIXED,
#     "name": "VGG16_Regression",
#     "level": const.SMOOTHNESS,
#     "model": const.VGG16,

# }

# vgg16_asphalt_regression_params = {
#     **global_config.global_config,
#     **default_params,
#     "batch_size": 16,
#     "epochs": 30,
#     "learning_rate": 0.00003,
#     "is_regression": True,
#     "eval_metric": const.EVAL_METRIC_ACCURACY,
#     "clm": False,
#     "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
#     "name": "VGG16_regression_asphalt",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.VGG16,

# }

# vgg16_asphalt_classification_params = {
#     **global_config.global_config,
#     **default_params,
#     "batch_size": 16,
#     "epochs": 30,
#     "learning_rate": 0.00003,
#     "is_regression": False,
#     "eval_metric": const.EVAL_METRIC_ACCURACY,
#     "clm": False,
#     "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
#     "name": "VGG16_classification_asphalt",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.VGG16,

# }

# vgg16_asphalt_CLM_params = {
#     **global_config.global_config,
#     **default_params,
#     "batch_size": 64,
#     "epochs": 20,
#     "learning_rate": 0.01,
#     "is_regression": False,
#     "eval_metric": const.EVAL_METRIC_ACCURACY,
#     "clm": True,
#     "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
#     "name": "VGG16_CLM_asphalt",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.VGG16_CLM,
# }

# vgg16_asphalt_Rosati_params = {
#     **global_config.global_config,
#     **default_params,
#     "batch_size": 64,
#     "epochs": 30,
#     "learning_rate": 0.0001,
#     "is_regression": False,
#     "eval_metric": const.EVAL_METRIC_ACCURACY,
#     "clm": True,
#     "two_optimizers":False,
#     "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
#     "name": "VGG16_CLM_asphalt",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.VGG16_CLM,
# }

# vgg16_asphalt_crophalf_regression_params = {
#     **global_config.global_config,
#     **default_params,
#     "transform": 
#         {"resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_HALF,
#         "normalize": const.NORM_DATA,},
#     "batch_size": 96,
#     "epochs": 20,
#     "learning_rate": 0.00003,
#     "is_regression": True,
#     "eval_metric": const.EVAL_METRIC_MSE,
#     "project": const.PROJECT_SMOOTHNESS_FIXED,
#     "name": "VGG16_Regression",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.VGG16,

# }

# effnet_asphalt_crophalf_regression_params = {
#     **global_config.global_config,
#     **default_params,
#     "transform": 
#         {"resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_HALF,
#         "normalize": const.NORM_DATA,},
#     "batch_size": 96,
#     "epochs": 20,
#     "learning_rate": 0.0001,
#     "is_regression": True,
#     "eval_metric": const.EVAL_METRIC_MSE,
#     "project": const.PROJECT_SMOOTHNESS_FIXED,
#     "name": "effnet_Regression",
#     "level": const.ASPHALT,
#     "selected_classes": global_config.global_config.get("selected_classes")[const.ASPHALT],
#     "dataset": "V12/annotated/asphalt",
#     "model": const.EFFNET_LINEAR,

# }


vgg16_flatten = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "flatten_model",
    "level": const.FLATTEN,
    "model": const.VGG16,
    "head": const.CLASSIFICATION,
    "eval_metric": const.EVAL_METRIC_ALL,
    "learning_rate": 0.00056,
    "hierarchy_method": None,
    #"fc_neurons": 512,
}

B_CNN = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "B_CNN_CLASSIFICATION_QWK",
    "level": const.HIERARCHICAL,
    "model": const.BCNN,
    "head": const.CLASSIFICATION_QWK, #'regression', 'classification', 'corn', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'use_condition_layer', 'b_cnn'
    "lw_modifier": True,
    "fc_neurons": 512,
}



C_CNN = {
    **global_config.global_config,
    **default_params,
    "batch_size": 128,
    "epochs": 1,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "C_CNN_classification",
    "level": const.HIERARCHICAL,
    "model": const.CCNN,
    "head": const.CLASSIFICATION, #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'None',
    "lw_modifier": False,
}

H_NET = {
    **global_config.global_config,
    **default_params,
    "batch_size": 128,
    "epochs": 1,
    "learning_rate": 0.0001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "HiearchyNet_classification",
    "level": const.HIERARCHICAL,
    "model": const.HNET,
    "head": const.CLASSIFICATION, #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'None',
    "lw_modifier": False,
}

GH_CNN = {
    **global_config.global_config,
    **default_params,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ACCURACY,
    "fine_eval_metric": const.EVAL_METRIC_ACCURACY,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "GH_CNN_CLM",
    "level": const.HIERARCHICAL,
    "model": const.GHCNN,
    "head": const.CLASSIFICATION_QWK, #'regression', 'classification', 'corn', 'clm', 'clm_kappa', 'classification_kappa',
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'use_condition_layer', 'b_cnn'
    "lw_modifier": True,
}

B_CNN_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.BCNN,
    "method": "grid",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP,
    "name": "B_CNN",
    "level": const.HIERARCHICAL,
    "head": const.CLASSIFICATION,
    "hierarchy_method": const.MODELSTRUCTURE,  
    "lw_modifier": True,
    "sweep_counts": 10,
}

C_CNN_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.CCNN,
    "method": "grid",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP,
    "name": "C_CNN_CLASSIFICATION",
    "level": const.HIERARCHICAL,
    "head": const.CLASSIFICATION,
    "hierarchy_method": const.MODELSTRUCTURE,  
    "lw_modifier": True,
    "sweep_counts": 10,
}