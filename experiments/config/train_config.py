from src import constants as const
from experiments.config  import global_config

default_params = {
    "batch_size": 16, #16,  # 48
    "epochs": 2,
    #"learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "eval_metric": const.EVAL_METRIC_ALL,
    'is_regression': False,
    #"max_class_size": None,
    "lr_scheduler": True,
    "freeze_convs": False,
    "gamma": 0.5,
}

default_search_params = {
    #"batch_size": {"values": [16, 48, 64]},
    "epochs": {"value": 12},
    "learning_rate": {"values": [0.001, 0.01, 0.1]},
    "gamma": {"values": [0.1, 0.5, 0.9]},
    #"learning_rate": {"distribution": "log_uniform_values", "min": 1e-05, "max": 0.001},
    #"optimizer": {"value": const.OPTI_ADAM},
    #"fc_neurons": {"values": [512, 1024]},
    #"freeze_convs": {"values": [True, False]}
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

vgg16_flatten = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_FINAL,
    "optimizer": const.OPTI_ADAM,
    "name": "fine_flatten",
    "level": const.FLATTEN,
    "model": const.VGG16,
    "head": const.CLASSIFICATION,
    "eval_metric": const.EVAL_METRIC_ALL,
    "learning_rate": 0.00005,
    "hierarchy_method": const.FLATTEN,
    #"fc_neurons": 512,
}


vgg16_surface_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_FINAL,
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
    "project": const.PROJECT_FINAL,
    "name": "CC_fine_CORN",
    "level": const.SMOOTHNESS,
    "model": const.VGG16,
    "learning_rate": 0.00005,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    "head": const.CORN,
    "hierarchy_method": const.CC,
    "eval_metric": const.EVAL_METRIC_ALL,
    # "gpu_kernel": 0,
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


B_CNN = {
    **global_config.global_config,
    **default_params,
    #"batch_size": 64,
    #"epochs": 12,
    "learning_rate": 0.005,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ALL,
    "fine_eval_metric": const.EVAL_METRIC_ALL,
    "project": const.PROJECT_FINAL,
    "name": "B_CNN_CLM",
    "level": const.HIERARCHICAL,
    "model": const.BCNN,
    "head": const.CLM, #'regression', 'classification', 'corn', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'use_condition_layer', 'b_cnn'
    "lw_modifier": True,
    #"lr_scheduler": True,
}


C_CNN = {
    **global_config.global_config,
    **default_params,
    "learning_rate": 0.001, 
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ALL,
    "fine_eval_metric": const.EVAL_METRIC_ALL,
    "project": const.PROJECT_FINAL,
    "name": "C_CNN_CLM",
    "level": const.HIERARCHICAL,
    "model": const.CCNN,
    "head": const.CLM, #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'None',
    "lw_modifier": True,
    #"lr_scheduler": True,
}

H_NET = {
    **global_config.global_config,
    **default_params,
    #"batch_size": 64,
    #"epochs": 12,
    "learning_rate": 0.01,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ALL,
    "fine_eval_metric": const.EVAL_METRIC_ALL,
    "project": const.PROJECT_FINAL,
    "name": "HiearchyNet_CLM",
    "level": const.HIERARCHICAL,
    "model": const.HNET,
    "head": const.CLM, #'regression', 'classification', 'obd', 'clm'
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'None',
    "lw_modifier": False,
    #"lr_scheduler": False,
}

GH_CNN = {
    **global_config.global_config,
    **default_params,
    #"batch_size": 64,
    #"epochs": 12,
    "learning_rate": 0.001,
    "optimizer": const.OPTI_ADAM,
    "coarse_eval_metric": const.EVAL_METRIC_ALL,
    "fine_eval_metric": const.EVAL_METRIC_ALL,
    "project": const.PROJECT_FINAL,
    "name": "GH_CNN_CLM",
    "level": const.HIERARCHICAL,
    "model": const.GHCNN,
    "head": const.CLM, #'regression', 'classification', 'corn', 'clm', 'clm_kappa', 'classification_kappa',
    "hierarchy_method": const.MODELSTRUCTURE, #'use_ground_truth', 'use_condition_layer', 'b_cnn'
    "lw_modifier": True,
    #"lr_scheduler": True,    
}


####### Sweeps for different hierarchical methods ########

B_CNN_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.BCNN,
    "method": "bayes",
    "learning_rate": 0.00001,
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_BCNN,
    "name": "B_CNN_CLASSIFICATION",
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
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_CCNN,
    "name": "C_CNN_CLASSIFICATION",
    "level": const.HIERARCHICAL,
    "head": const.CLM,
    "hierarchy_method": const.MODELSTRUCTURE,  
    "lw_modifier": True,
    "sweep_counts": 10,
}

H_NET_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.HNET,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_HNET,
    "name": "H_NET_CLASSIFICATION",
    "level": const.HIERARCHICAL,
    "head": const.CLASSIFICATION,
    "hierarchy_method": const.MODELSTRUCTURE,  
    "lw_modifier": True,
    "sweep_counts": 10,
}

GH_CNN_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.GHCNN,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_GHCNN,
    "name": "GH_CNN_CLASSIFICATION",
    "level": const.HIERARCHICAL,
    "head": const.CLASSIFICATION,
    "hierarchy_method": const.MODELSTRUCTURE,  
    "lw_modifier": True,
    "sweep_counts": 10,
}

####### Sweeps for different 'heads'/ ordinal methods ########

asphalt_quality_classification_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.VGG16,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {**default_search_params,                 
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_CLASSIFICATION,
    "name": "Quality_Asphalt_Classification",
    "level": const.ASPHALT,
    "head": const.CLASSIFICATION,
    "sweep_counts": 10,
    "hierarchy_method": const.FLATTEN, 
}

asphalt_quality_regression_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.VGG16,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {'epochs': {'value': 12},
                      'learning_rate': {'values': [5e-05, 1e-04, 5e-04]},  
                      'gamma': {'values': [0.1, 0.5, 0.9]},                     
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_REGRESSION,
    "name": "Quality_Asphalt_Regression",
    "level": const.ASPHALT,
    "head": const.REGRESSION,
    "sweep_counts": 10,
    "hierarchy_method": const.FLATTEN, 
}

asphalt_quality_corn_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.VGG16,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {'epochs': {'value': 12},
                      'learning_rate': {'values': [5e-05, 1e-04, 5e-04]},  
                      'gamma': {'values': [0.1, 0.5, 0.9]},                   
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_CORN,
    "name": "Quality_Asphalt_CORN",
    "level": const.ASPHALT,
    "head": const.CORN,
    "sweep_counts": 10,
    "hierarchy_method": const.FLATTEN, 
}

asphalt_quality_fixed_params = {
    **global_config.global_config,
    **default_params,
    "project": const.PROJECT_MULTI_LABEL_FIXED,
    "name": "Quality_Asphalt_CLM",
    "level": const.ASPHALT,
    "model": const.VGG16,
    "learning_rate": 0.01,
    #"dataset": "V1_0/train",
    #"metadata": "V1_0/metadata",
    #"train_valid_split_list": "train_valid_split.csv",
    "head": const.CLM,
    "hierarchy_method": const.FLATTEN,
    "eval_metric": const.EVAL_METRIC_ALL,
    # "gpu_kernel": 0,
}


asphalt_quality_clm_sweep_params = {
    **global_config.global_config,
    **default_params,
    'model': const.VGG16,
    "method": "bayes",
    "metric": {"name": "eval/accuracy/fine", "goal": "maximize"},
    "search_params": {'epochs': {'value': 12},
                      'learning_rate': {'values': [0.1, 0.01, 0.001]},  
                      'gamma': {'values': [0.1, 0.5, 0.9]},              
                     },
    "project": const.PROJECT_MULTI_LABEL_SWEEP_CLM,
    "name": "Quality_Asphalt_CLM",
    "level": const.ASPHALT,
    "head": const.CLM,
    "sweep_counts": 10,
    "hierarchy_method": const.FLATTEN, 
}

vgg16_asphalt_clm_params = {
    **global_config.global_config,
    **default_params,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 0.00003,
    "eval_metric": const.EVAL_METRIC_ALL,
    "project": const.PROJECT_ORDINAL_REGRESSION_FIXED,
    "name": "VGG16_clm_asphalt",
    "level": const.ASPHALT,
    "model": const.VGG16,
    "head": const.CLM,
    "hierarchy_method": const.FLATTEN,
}