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

CC_V1_0 = {
    **global_config.global_config,
    "name": "effnet_surface_quality_prediction",
    "model_dict": {
        "trained_model": "surface-efficientNetV2SLinear-20240610_185408-j3ob3p5o_epoch6.pt",
        # with blur augmentation
        # "trained_model": "surface-efficientNetV2SLinear-20240614_120022-k5dko7ax_epoch10.pt",
        # with higher blur augmentation
        # "trained_model": "surface-efficientNetV2SLinear-20240617_105759-nh5vboqz_epoch16.pt",
        "level": const.TYPE,
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240610_235740-h8r4ubgv_epoch17.pt",
                # with blur augmentation
                # "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240617_140426-orcs8ch9_epoch16.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240611_003637-o35q8be6_epoch12.pt",
                # with blur augmentation
                # "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240617_143238-mbbiz38y_epoch18.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240611_011325-t3y17ezx_epoch17.pt",
                # with blur augmentation
                # "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240617_144050-2t761f6f_epoch17.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240611_021327-s77xoig3_epoch15.pt",
                # with blur augmentation
                # "trained_model": "smoothness-sett-efficientNetV2SLinear-20240617_145801-wjlhdxyh_epoch7.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240611_032059-19660tlf_epoch17.pt",
                # with blur augmentation
                # "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240617_150619-derdvtjr_epoch8.pt",
                "level": const.QUALITY,
            },
        },
    },
    "root_data": str(global_config.ROOT_DIR / "data"),
    # "dataset": "weseraue/imgs_2048",
    # "dataset": "weseraue/original",
    # "dataset": "weseraue/paving_stones",
    # "dataset": "V1_0/s_1024",
    # "dataset": "lndw",
    "dataset": "berlin/vset_all",
    "transform": {
        "resize": (384, 384),
        # "crop": const.CROP_LOWER_MIDDLE_HALF_PANO,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        # "crop": "small_pano",
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "gpu_kernel": 1,
    "batch_size": 16,
}

blur_V1_0 = {
    **global_config.global_config,
    "name": "effnet_blur_surface_pred",
    "model_dict": {
        "trained_model": "",
        "level": const.TYPE,
    },
    "dataset": "V1_0/annotated",
    "transform": {
        "preresize": (96, 96),
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        # "gaussian_blur_kernel": 11,
        # "gaussian_blur_sigma": 5,
        # "gaussian_blur_fixed": True,
    },
    "gpu_kernel": 1,
    "batch_size": 16,
}

blur_weseraue = {
    **global_config.global_config,
    "name": "effnet_blur_surface_pred_weseraue",
    "model_dict": {
        "trained_model": "",
        "level": const.TYPE,
    },
    "root_data": str(global_config.ROOT_DIR / "data"),
    "dataset": "weseraue/paving_stones",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF_PANO,
        # "crop": const.CROP_SMALL_PANO,
        # "crop": const.CROP_SUPER_SMALL_PANO,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
        # "gaussian_blur_fixed": True,
    },
    "gpu_kernel": 0,
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

effnet_surface = {
    **global_config.global_config,
    "name": "effnet_surface_quality_prediction",
    "model_dict": {
        "trained_model": "surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt",
        "level": const.TYPE,
    },
    "dataset": "V103/unsorted_images",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "gpu_kernel": 1,
    "save_state": True,
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
    # "model_dict": {"trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt"},
    "model_dict": {
        "trained_model": "surface-efficientNetV2SLinear-20240610_185408-j3ob3p5o_epoch6.pt"
    },
    # "dataset": "V9/annotated",
    # "dataset": "V9/metadata/model_predictions/misclassified_images/surface",
    "root_data": str(global_config.ROOT_DIR / "data"),
    "dataset": "lndw",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "gpu_kernel": 1,
    "batch_size": 16,
}

cam_surface_weseraue = {
    **global_config.global_config,
    "name": "cam_surface_prediction",
    # "model_dict": {"trained_model": "surface-efficientNetV2SLinear-20240704_211831-ntvzab3t_epoch7.pt"}, # no blur run 5
    "model_dict": {
        "trained_model": "surface-efficientNetV2SLinear-20240704_220436-7v8p5y2o_epoch8.pt"
    },  # noblur run 6
    "root_data": str(global_config.ROOT_DIR / "data"),
    "dataset": "weseraue_cam_analysis",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF_PANO,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "gpu_kernel": 1,
    "batch_size": 16,
}

effnet_scenery = {
    **global_config.global_config,
    "name": "effnet_scenery_prediction",
    "model_dict": {
        # "trained_model": "flatten-efficientNetV2SLinear-20240613_095915-81nd24pa_epoch12.pt",
        # "trained_model": "flatten-efficientNetV2SLinear-20240613_103053-033v1uet_epoch17.pt",
        # "trained_model": "flatten-efficientNetV2SLinear-20240711_092043-n8x73ojw_epoch17.pt",
        # "trained_model": "flatten-efficientNetV2SLinear-20240717_175624-hbucwrbc_epoch10.pt",
        # "trained_model": "flatten-efficientNetV2SLinear-20240718_091851-6t2bdijv_epoch9.pt", # avg pool 6
        "trained_model": "flatten-efficientNetV2SLinear-20240718_125004-2szn5maz_epoch10.pt",  # crop None
        "level": "scenery",
    },
    "dataset": "road_scenery",
    # "root_data": str(global_config.ROOT_DIR / "data"),
    # "dataset": "berlin/vset_all",
    # "dataset": "V1_0/annotated",
    # "dataset": "V12/annotated/no_street",
    # "dataset": "V12/annotated/not_recognizable",
    "transform": {
        "resize": (384, 384),
        # "crop": const.CROP_LOWER_HALF,
        "crop": None,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 16,
    "gpu_kernel": 1,
}

resnet_test = {
    **global_config.global_config,
    "name": "test_resnet_prediction",
    "model_dict": {
        "trained_model": "surface-resnet50-20250101_104323_epoch4.pt",
        "level": "surface",
    },
    "dataset": "test_images",
    "transform": {
        "resize": (256, 256),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.TEST_IMAGES_MEAN, const.TEST_IMAGES_SD),
    },
    "batch_size": 16,
}
