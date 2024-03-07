from src import constants as const
from experiments.config import global_config

rateke_CC = {
    **global_config.global_config,
    "name": "test_RatekeCNN_VGG16_prediction",
    "model_dict": {
        "trained_model": "surface-rateke-20240207_202104-gnzhpn11_epoch0.pt",
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-vgg16-20240207_202414-krno5gva_epoch0.pt"
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-vgg16-20240207_202524-jqetza3o_epoch0.pt"
            },
        },
    },
    "dataset": "V0/predicted",
    "transform": {
        "resize": const.H256_W256,
        # "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V4_ANNOTATED_MEAN, const.V4_ANNOTATED_SD),
    },
    "batch_size": 48,
}

vgg16_surface = {
    **global_config.global_config,
    "name": "surface_prediction",
    "model_dict": {"trained_model": "surface-vgg16-20240215_122253-wgch26j7_epoch18.pt"},
    "dataset": "V5_c3",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
    },
    "batch_size": 96,
}

cam_surface = {
    **global_config.global_config,
    "name": "cam_surface_prediction",
    # "model_dict": {"trained_model": "surface-vgg16-20240215_122253-wgch26j7_epoch18.pt"},
    "model_dict": {"trained_model": "surface-rateke-20240207_202104-gnzhpn11_epoch0.pt"},
    "dataset": "CAM",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
    },
    "batch_size": 8,
}