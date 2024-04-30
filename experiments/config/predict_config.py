from src import constants as const
from experiments.config import global_config
import os

segment_color = {
    const.SEGMENT_ROAD: (255, 0, 0), # red
    const.SEGMENT_BIKE: (0, 255, 0), # green
    const.SEGMENT_SIDEWALK: (0, 0, 255), # blue
    const.SEGMENT_CROSSWALK: (255, 255, 0), # yellow
    'construction--flat--curb-cut': (255, 0, 255), # pink
    'construction--flat--driveway': (255, 0, 255), # pink
    'construction--flat--parking': (255, 0, 255), # pink
    'construction--flat--parking-aisle': (255, 0, 255), # pink
    'construction--flat--pedestrian-area': (0, 127, 255), # light blue
    'construction--flat--rail-track': (255, 0, 255), # pink
    'construction--flat--road-shoulder': (255, 0, 255), # pink
    'construction--flat--service-lane': (255, 0, 255), # pink
    'construction--flat--traffic-island': (255, 0, 255), # pink
    'void--ground': (0, 255, 255), # turquoise
    'void--unlabeled': (0, 255, 255), # turquoise
    'void--static': (0, 255, 255), # turquoise
    'nature--beach': (127, 0, 255), # lilac
    'nature--sand': (127, 0, 255), # lilac
    'nature--snow': (127, 0, 255), # lilac
    'nature--terrain': (127, 0, 255), # lilac
    # 'nature--vegetation': (127, 0, 255), # lilac
    # 'nature--water': (127, 0, 255), # lilac
}

train_validation_segmentation_CC = {
    **global_config.global_config,
    "name": "effnet_surface_prediction",
    "model_dict": {
       # "trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt",
        "trained_model": "surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt", 
        # "submodels": {
        #     const.ASPHALT: {
        #         "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240314_202655-x67n9qjz_epoch18.pt"
        #     },
        #     const.CONCRETE: {
        #         "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240314_221414-z9pumhri_epoch18.pt"
        #     },
        #     const.PAVING_STONES: {
        #         "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240314_223314-c8cxtraf_epoch14.pt"
        #     },
        #     const.SETT: {
        #         "trained_model": "smoothness-sett-efficientNetV2SLinear-20240314_233003-mplaq0xd_epoch19.pt"
        #     },
        #     const.UNPAVED: {
        #         "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240315_001707-zu6wt2fs_epoch16.pt"
        #     },
        # },
    },
    "dataset": "V11/annotated",
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "segment_color": segment_color,
    # 'mapillary_token_path': os.path.join(const.ROOT_DIR, 'mapillary_token.txt'),
    'segmentation_folder': 'V11/segmentation',
    "saving_postfix": "detection",
    'segmentation': 'crop',
    # 'segmentation': 'mask',
    'segmentation': 'mask_crop',
    'gpu_kernel': 0,
}

train_validation_segmentation_CC_v2 = {
    **global_config.global_config,
    "name": "effnet_surface_pred_segment",
    "model_dict": {
       # "trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt",
        "trained_model": "surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt", 
        # "submodels": {
        #     const.ASPHALT: {
        #         "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240314_202655-x67n9qjz_epoch18.pt"
        #     },
        #     const.CONCRETE: {
        #         "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240314_221414-z9pumhri_epoch18.pt"
        #     },
        #     const.PAVING_STONES: {
        #         "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240314_223314-c8cxtraf_epoch14.pt"
        #     },
        #     const.SETT: {
        #         "trained_model": "smoothness-sett-efficientNetV2SLinear-20240314_233003-mplaq0xd_epoch19.pt"
        #     },
        #     const.UNPAVED: {
        #         "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240315_001707-zu6wt2fs_epoch16.pt"
        #     },
        # },
    },
    "dataset": "V11/annotated",
    "transform": {
        "resize": (384, 384),
        # "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "segment_color": segment_color,
    # 'mapillary_token_path': os.path.join(const.ROOT_DIR, 'mapillary_token.txt'),
    'segmentation_folder': 'V11/segmentation',
    "saving_postfix": "detection",
    # 'segmentation': 'crop',
    # 'segmentation': 'mask',
    'seg_mask_style': 'outside_blur', # None
    # 'seg_mask_style': 'outside_blur_convex', # allways convex
    'seg_crop_style': 'segmentation', # const.CROP_LOWER_MIDDLE_HALF
    'seg_pre_crop': const.CROP_LOWER_MIDDLE_HALF,
    'segmentation_selection_func': 'seg_sel_func_max_area_in_lower_half_crop',
    'seg_threshold': 0.05,
    'gpu_kernel': 0,
}

seg_analysis = {
    **global_config.global_config,
    "name": "effnet_surface_pred_segment",
    "model_dict": {
       # "trained_model": "surface-efficientNetV2SLinear-20240314_164055-mi0872lh_epoch6.pt",
        "trained_model": "surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt", 
    },
    "dataset": "seg_analysis/original/not_recognizable/multi/park_path",
    "transform": {
        "resize": (384, 384),
        # "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "segment_color": segment_color,
    # 'mapillary_token_path': os.path.join(const.ROOT_DIR, 'mapillary_token.txt'),
    'segmentation_folder': 'seg_analysis/segmentation/not_recognizable/multi/park_path',
    "saving_postfix": "detection",
    # 'segmentation': 'crop',
    # 'segmentation': 'mask',
    'seg_mask_style': 'outside_blur', # None
    # 'seg_mask_style': 'outside_blur_convex', # allways convex
    'seg_crop_style': 'segmentation', # const.CROP_LOWER_MIDDLE_HALF
    'seg_pre_crop': const.CROP_LOWER_MIDDLE_HALF,
    'segmentation_selection_func': 'seg_sel_func_all',
    'seg_threshold': 0.01,
    'gpu_kernel': 0,
}

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
    "dataset": "tmp",
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

segmentation_CC_test = {
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
    'segment_color': segment_color,
    # 'mapillary_token_path': os.path.join(const.ROOT_DIR, 'mapillary_token.txt'),
}

# segmentation = {
#     **global_config.global_config,
#     # "mode": "testing",
#     "dataset": "segmentation/original",
#     "segment_color": segment_color,
#     # 'mapillary_token_path': os.path.join(const.ROOT_DIR, 'mapillary_token.txt'),
# }