from src import constants as const
from experiments.config import global_config



# Models:
# B_CNN_CORN_GT: hierarchical-B_CNN-corn-use_ground_truth-20241031_005138-32tlqlfa_epoch11.pt


vgg16_surface = {
    **global_config.global_config,
    "name": "surface_prediction",
    "model_dict": {"trained_model": "surface-vgg16-20240202_125044-1uv8oow5.pt"},
    "dataset": "V5_c3",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
}

B_CNN = {
    **global_config.global_config,
    "name": "B_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-xu274wad_epoch2.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

C_CNN = {
    **global_config.global_config,
    "name": "C_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-Condition_CNN-classification-use_model_structure-20241027_180014-ymnv1106_epoch11.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

H_NET = {
    **global_config.global_config,
    "name": "H_NET_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-HiearchyNet-classification-use_model_structure-20241027_180036-ob5xv766_epoch8.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

GH_CNN = {
    **global_config.global_config,
    "name": "GH_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-GH_CNN-classification-use_model_structure-20241027_180609-qc6an27g_epoch1.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

B_CNN_CORN_GT = {
    **global_config.global_config,
    "name": "B_CNN_CORN_GT_prediction",
    "model_dict": {"trained_model": "final\\hierarchical-B_CNN-corn-use_ground_truth-20241031_005138-32tlqlfa_epoch11.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": "V1_0",
    "metadata": "streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

CC = {
    **global_config.global_config,
    "name": "CC_Classification_prediction",
    "model_dict": {
        "trained_model": "surface-vgg16-classification-CC-20241103_102134_epoch0.pt",
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-vgg16-classification-CC-20241103_102236_epoch0.pt"
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-vgg16-classification-CC-20241103_102330_epoch0.pt"
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-vgg16-classification-CC-20241103_102422_epoch0.pt"
            },
            const.SETT: {
                "trained_model": "smoothness-sett-vgg16-classification-CC-20241103_102514_epoch0.pt"
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-vgg16-classification-CC-20241103_102612_epoch0.pt"
            },
        },
    },
    "dataset": "V1_0",
    "metadata": "streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        # "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": False,
}



# B_CNN = {
#     **global_config.global_config,
#     "name": "B_CNN_prediction",
#     "model_dict": {"trained_model": "multilabel-BCNN_pretrained-20240505_133427-c549if0b_epoch39.pt"}, 
#     "dataset": "V11/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "batch_size": 96,
#     "save_features": True
# }
# B_CNN = {
#     **global_config.global_config,
#     "name": "B_CNN_prediction",
#     "model_dict": {"trained_model": "multilabel-BCNN_pretrained-20240505_133427-c549if0b_epoch39.pt"}, 
#     "dataset": "V11/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "batch_size": 96,
#     "save_features": True
# }


# C_CNN_PRE = {
#     **global_config.global_config,
#     "name": "C_CNN_prediction",
#     "model_dict": {"trained_model": "hierarchical-Condition_CNN_CLM_PRE-20240820_163509_epoch0.pt"}, 
#     "dataset": "V12/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "level": const.HIERARCHICAL,
#     "head": 'regression', #'regression', 'classification', 'obd', 'clm'
#     "hierarchy_method": const.GROUNDTRUTH, #'use_ground_truth', 'use_condition_layer', 'top_coarse_prob'
#     "batch_size": 96,
#     "save_features": False
# }
