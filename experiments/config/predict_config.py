from src import constants as const
from experiments.config import global_config

CC = {
    **global_config.global_config,
    "name": "all_train_effnet_surface_quality_prediction",
    "model_dict": {
       # "trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt",
        "trained_model": "surface-efficientNetV2SLinear-20240318_114422-a68tf9lt_epoch4.pt",
        "level": const.TYPE,
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240314_202655-x67n9qjz_epoch18.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240314_221414-z9pumhri_epoch18.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240314_223314-c8cxtraf_epoch14.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240314_233003-mplaq0xd_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240315_001707-zu6wt2fs_epoch16.pt",
                "level": const.QUALITY,
            },
        },
    },
    "dataset": "V9/annotated",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V9_ANNOTATED_MEAN, const.V9_ANNOTATED_SD),
    },
    "batch_size": 16,
}

all_train_CC = {
    **global_config.global_config,
    "name": "all_train_effnet_surface_quality_prediction",
    "model_dict": {
       # "trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt",
        "trained_model": "surface-efficientNetV2SLinear-20240318_114422-a68tf9lt_epoch4.pt",
        "level": const.TYPE,
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240314_202655-x67n9qjz_epoch18.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240314_221414-z9pumhri_epoch18.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240314_223314-c8cxtraf_epoch14.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240314_233003-mplaq0xd_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240315_001707-zu6wt2fs_epoch16.pt",
                "level": const.QUALITY,
            },
        },
    },
    "dataset": "V9/annotated",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V9_ANNOTATED_MEAN, const.V9_ANNOTATED_SD),
    },
    "batch_size": 16,
}

V11_type_V9_quality_CC = {
    **global_config.global_config,
    "name": "effnet_surface_quality_prediction",
    "model_dict": {
        "trained_model": "surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt",
        "level": const.TYPE, 
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240314_202655-x67n9qjz_epoch18.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240314_221414-z9pumhri_epoch18.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240314_223314-c8cxtraf_epoch14.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240314_233003-mplaq0xd_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240315_001707-zu6wt2fs_epoch16.pt",
                "level": const.QUALITY,
            },
        },
    },
    "root_data": str(global_config.ROOT_DIR / "data"),
    "root_predict": str(global_config.ROOT_DIR / "data" / "prediction"),
    "dataset": "weseraue/imgs_1024",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "gpu_kernel": 1,
}

vgg16_surface = {
    **global_config.global_config,
    "name": "surface_prediction",
    "model_dict": {
        "trained_model": "surface-vgg16-20240215_122253-wgch26j7_epoch18.pt",
        "level": const.TYPE,
        },
    "dataset": "V5_c5",
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
    # "model_dict": {"trained_model": "surface-efficientNetV2SLinear-20240312_090721-iia9tei2_epoch14.pt"},
    "model_dict": {"trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt"},
    # "dataset": "V9/annotated",
    "dataset": "V9/metadata/model_predictions/misclassified_images/surface",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V9_ANNOTATED_MEAN, const.V9_ANNOTATED_SD),
    },
    "batch_size": 8,
}