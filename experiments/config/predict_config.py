from src import constants as const
from experiments.config import global_config

model_dict = {
    "trained_model": "surface-efficientNetV2SLinear-20240610_185408-j3ob3p5o_epoch6.pt",
    "level": const.TYPE,
    "submodels": {
        const.ASPHALT: {
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240610_235740-h8r4ubgv_epoch17.pt",
            "level": const.QUALITY,
        },
        const.CONCRETE: {
            "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240611_003637-o35q8be6_epoch12.pt",
            "level": const.QUALITY,
        },
        const.PAVING_STONES: {
            "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240611_011325-t3y17ezx_epoch17.pt",
            "level": const.QUALITY,
        },
        const.SETT: {
            "trained_model": "smoothness-sett-efficientNetV2SLinear-20240611_021327-s77xoig3_epoch15.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240611_032059-19660tlf_epoch17.pt",
            "level": const.QUALITY,
        },
    },
}

default_params = {
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
}

CC_V1_0_train = {
    **global_config.global_config,
    **default_params,
    "name": "efficientnet_CC_prediction",
    "model_dict": model_dict,
    "dataset": "V1_0/train",
}

CC_V1_0_test = {
    **global_config.global_config,
    **default_params,
    "name": "efficientnet_CC_prediction",
    "model_dict": model_dict,
    "dataset": "V1_0/test",
}

CC_V1_0_RTK = {
    **global_config.global_config,
    **default_params,
    "name": "efficientnet_CC_prediction",
    "model_dict": model_dict,
    "dataset": "RTK",
}
