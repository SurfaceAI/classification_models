### hint: last part should be 'normalization values', this is extended by calculation of new dataset normalization

import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

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

# smoothness types
EXCELLENT = "excellent"
GOOD = "good"
INTERMEDIATE = "intermediate"
BAD = "bad"
VERY_BAD = "very_bad"
HORRIBLE = "horrible"

SMOOTHNESS_INT = {
    EXCELLENT: 1,
    GOOD: 2,
    INTERMEDIATE: 3,
    BAD: 4,
    VERY_BAD: 5,
    HORRIBLE: 6,
}

# classification level
SURFACE = "surface"
SMOOTHNESS = "smoothness"
FLATTEN = "flatten"

# dataset
V0 = "V0"
V1 = "V1"
V2 = "V2"
V3 = "V3"
V4 = "V4"

# label type
ANNOTATED = "annotated"

# project names
PROJECT_SURFACE_FIXED = "road-surface-classification-type"
PROJECT_SURFACE_SWEEP = "sweep-road-surface-classification-type"
PROJECT_SMOOTHNESS_FIXED = "road-surface-classification-quality"
PROJECT_SMOOTHNESS_SWEEP = "sweep-road-surface-classification-quality"
PROJECT_FLATTEN_FIXED = "road-surface-classification-flatten"
PROJECT_FLATTEN_SWEEP = "sweep-road-surface-classification-flatten"


# model names
EFFICIENTNET = "efficientNetV2SLogsoftmax"
EFFNET_LINEAR = "efficientNetV2SLinear"
VGG16 = "vgg16"
RATEKE = "rateke"
# temporay onla
VGG16REGRESSION = "vgg16Regression"

# sweep names

# architecture
# EFFICIENTNET_V2_S = "Efficient Net v2 s"

# optimizer
OPTI_ADAM = "adam"

# evaluation metrics
EVAL_METRIC_ACCURACY = "acc"
EVAL_METRIC_MSE = "MSE"

# checkpoint & early stopping
CHECKPOINT_DEFAULT_TOP_N = 1
EARLY_STOPPING_DEFAULT = np.Inf

### segmentation
# mapillary road types
SEGMENT_ROAD = 'construction--flat--road'
SEGMENT_BIKE = 'construction--flat--bike-lane'
SEGMENT_SIDEWALK = 'construction--flat--sidewalk'
SEGMENT_CROSSWALK = 'construction--flat--crosswalk-plain'

### preprocessing
# image size
H256_W256 = (256, 256)

# crop
CROP_LOWER_MIDDLE_THIRD = "lower_middle_third"
CROP_LOWER_MIDDLE_HALF = "lower_middle_half"

# normalization
NORM_IMAGENET = "imagenet"
NORM_DATA = "from_data"

# normalization values
IMAGNET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_SD = [0.229, 0.224, 0.225]



V4_ANNOTATED_MEAN = [0.4205051362514496, 0.439302921295166, 0.42983368039131165]
V4_ANNOTATED_SD = [0.23013851046562195, 0.23709532618522644, 0.2626153826713562]

V6_ANNOTATED_MEAN = [0.4241970479488373, 0.4434114694595337, 0.4307621419429779]
V6_ANNOTATED_SD = [0.22932860255241394, 0.23517057299613953, 0.26160895824432373]

V7_ANNOTATED_ASPHALT_MEAN = [0.41713881492614746, 0.4449938237667084, 0.44482147693634033]
V7_ANNOTATED_ASPHALT_SD = [0.23219026625156403, 0.24139828979969025, 0.2703194320201874]

V9_ANNOTATED_MEAN = [0.42359083890914917, 0.4423845410346985, 0.4317628741264343]
V9_ANNOTATED_SD = [0.2301570326089859, 0.23571978509426117, 0.26122480630874634]
