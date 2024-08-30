from src import constants as const
from experiments.config import global_config

# TODO: insert trained model files
model_dict = {
    "trained_model": "surface-efficientNetV2SLinear-20240805_142047-5bvhwua5_epoch6.pt",
    "level": const.TYPE,
    "submodels": {
        const.ASPHALT: {
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240805_150657-vbh3etku_epoch19.pt",
            "level": const.QUALITY,
        },
        const.CONCRETE: {
            "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240805_153157-6lg00blf_epoch16.pt",
            "level": const.QUALITY,
        },
        const.PAVING_STONES: {
            "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240805_153933-sag52zdy_epoch17.pt",
            "level": const.QUALITY,
        },
        const.SETT: {
            "trained_model": "smoothness-sett-efficientNetV2SLinear-20240805_155449-jj2sy16u_epoch18.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240805_160525-bfdgx097_epoch19.pt",
            "level": const.QUALITY,
        },
    },
}

CC_V1_0_train = {
    **global_config.global_config,
    "name": "train_CC_prediction",
    "model_dict": model_dict,
    "dataset": "V1_0/train",
}

CC_V1_0_test = {
    **global_config.global_config,
    "name": "test_CC_prediction",
    "model_dict": model_dict,
    "dataset": "V1_0/test",
}
