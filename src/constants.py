### hint: last part should be 'normalization values', this is extended by calculation of new dataset normalization

import numpy as np

# weights and biases
WANDB_MODE_ON = "online"
WANDB_MODE_OFF = "offline"
WANDB_DEFAULT_SWEEP_COUNTS = 10

# surface types
ASPHALT = "asphalt"
CONCRETE = "concrete"
SETT = "sett"
UNPAVED = "unpaved"
PAVING_STONES = "paving_stones"
PAVED = "paved"

# smoothness types
EXCELLENT = "excellent"
GOOD = "good"
INTERMEDIATE = "intermediate"
REGULAR = "regular"
BAD = "bad"
VERY_BAD = "very_bad"
HORRIBLE = "horrible"

SMOOTHNESS_INT = {
    EXCELLENT: 1,
    GOOD: 2,
    INTERMEDIATE: 3,
    REGULAR: 3,
    BAD: 4,
    VERY_BAD: 5,
    HORRIBLE: 6,
}

# classification level
SURFACE = "surface"
SMOOTHNESS = "smoothness"
FLATTEN = "flatten"

TYPE = "type"
QUALITY = "quality"

# project names
PROJECT_SURFACE_FIXED = "road-surface-classification-type"
PROJECT_SURFACE_SWEEP = "sweep-road-surface-classification-type"
PROJECT_SMOOTHNESS_FIXED = "road-surface-classification-quality"
PROJECT_SMOOTHNESS_SWEEP = "sweep-road-surface-classification-quality"

# model names
EFFNET_LINEAR = "efficientNetV2SLinear"

# architecture TODO
# EFFICIENTNET_V2_S = "Efficient Net v2 s"

# optimizer
OPTI_ADAM = "adam"

# evaluation metrics
EVAL_METRIC_ACCURACY = "acc"
EVAL_METRIC_MSE = "MSE"

# checkpoint & early stopping
CHECKPOINT_DEFAULT_TOP_N = 1
EARLY_STOPPING_DEFAULT = np.Inf # TODO

### preprocessing
# image size
H256_W256 = (256, 256)
H384_W384 = (384, 384)

# crop TODO!
# CROP_LOWER_MIDDLE_THIRD = "lower_middle_third"
CROP_LOWER_MIDDLE_HALF = "lower_middle_half"
# CROP_LOWER_MIDDLE_HALF_PANO = "lower_middle_half_pano"
# CROP_LOWER_HALF = "lower_half"
# CROP_SMALL_PANO = "small_pano"
# CROP_SUPER_SMALL_PANO = "super_small_pano"
CROP_LOWER_HALF_RTK = "lower_half_rtk"

# normalization
NORM_IMAGENET = "imagenet"
NORM_DATA = "from_data"

# normalization values
IMAGNET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_SD = [0.229, 0.224, 0.225]

V1_0_ANNOTATED_MEAN = [0.42834484577178955, 0.4461250305175781, 0.4350937306880951]
V1_0_ANNOTATED_SD = [0.22991590201854706, 0.23555299639701843, 0.26348039507865906]
