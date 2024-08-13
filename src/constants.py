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

# smoothness types
EXCELLENT = "excellent"
GOOD = "good"
INTERMEDIATE = "intermediate"
BAD = "bad"
VERY_BAD = "very_bad"
HORRIBLE = "horrible"

# SMOOTHNESS_INT = {
#     EXCELLENT: 1,
#     GOOD: 2,
#     INTERMEDIATE: 3,
#     BAD: 4,
#     VERY_BAD: 5,
#     HORRIBLE: 6,
# }

SMOOTHNESS_INT = {
    EXCELLENT: 0,
    GOOD: 1,
    INTERMEDIATE: 2,
    BAD: 3,
    VERY_BAD: 4,
    HORRIBLE: 5,
}


ASPHALT_EXCELLENT = "asphalt__excellent"
ASPHALT_GOOD = "asphalt__good"
ASPHALT_INTERMEDIATE = "asphalt__intermediate"
ASPHALT_BAD = "asphalt__bad"
CONCRETE_EXCELLENT = "concrete__excellent"
CONCRETE_GOOD = "concrete__good"
CONCRETE_INTERMEDIATE = "concrete__intermediate"
CONCRETE_BAD = "concrete__bad"
PAVING_STONES_EXCELLENT = "paving_stones__excellent"
PAVING_STONES_GOOD = "paving_stones__good"
PAVING_STONES_INTERMEDIATE = "paving_stones__intermediate"
PAVING_STONES_BAD = "paving_stones__bad"
SETT_GOOD = "sett__good"
SETT_INTERMEDIATE = "sett__intermediate"
SETT_BAD = "sett__bad"
UNPAVED_INTERMEDIATE = "unpaved__intermediate"
UNPAVED_BAD = "unpaved__bad"
UNPAVED_VERY_BAD = "unpaved__very_bad"


FLATTENED_INT = {
    ASPHALT_EXCELLENT: 0,
    ASPHALT_GOOD: 1,
    ASPHALT_INTERMEDIATE: 2,
    ASPHALT_BAD: 3,
    CONCRETE_EXCELLENT: 4,
    CONCRETE_GOOD: 5, 
    CONCRETE_INTERMEDIATE: 6, 
    CONCRETE_BAD: 7, 
    PAVING_STONES_EXCELLENT: 8,
    PAVING_STONES_GOOD: 9, 
    PAVING_STONES_INTERMEDIATE: 10, 
    PAVING_STONES_BAD: 11, 
    SETT_GOOD: 12, 
    SETT_INTERMEDIATE: 13, 
    SETT_BAD: 14, 
    UNPAVED_INTERMEDIATE: 15, 
    UNPAVED_BAD: 16, 
    UNPAVED_VERY_BAD: 17,
}

# classification level
SURFACE = "surface"
SMOOTHNESS = "smoothness"
FLATTEN = "flatten"
HIERARCHICAL = "hierarchical"

# dataset
V0 = "V0"
V1 = "V1"
V2 = "V2"
V3 = "V3"
V4 = "V4"
V10 = "V10"
V11 = "V11"
V12 = "V12"


# label type
ANNOTATED = "annotated"

# project names
PROJECT_SURFACE_FIXED = "road-surface-classification-type"
PROJECT_SURFACE_SWEEP = "sweep-road-surface-classification-type"
PROJECT_SMOOTHNESS_FIXED = "road-surface-classification-quality"
PROJECT_SMOOTHNESS_SWEEP = "sweep-road-surface-classification-quality"
PROJECT_FLATTEN_FIXED = "road-surface-classification-flatten"
PROJECT_FLATTEN_SWEEP = "sweep-road-surface-classification-flatten"
PROJECT_MULTI_LABEL_FIXED = "road-surface-classification-multi-label"
PROJECT_ORDINAL_REGRESSION_FIXED = "road-surface-classification-ordinal-regression"



# model names
EFFICIENTNET = "efficientNetV2SLogsoftmax"
EFFNET_LINEAR = "efficientNetV2SLinear"
VGG16 = "vgg16"
RATEKE = "rateke"
VGG16_CLM = "vgg16_CLM"
# temporay onla
VGG16REGRESSION = "vgg16Regression"
BCNN = "BCNN"
BCNN_PRE = "VGG16_B_CNN_PRE"
BCNNREGRESSION = "BCNNRegression"
CCNN = "ConditionCNN"
CCNN_PRE = "ConditionCNN_pretrained"
CCNNREGRESSION = "CCNNRegression"
CCNNCLMPRE = "Condition_CNN_CLM_PRE"
HNET = "HiearchyNet"
HNET_PRE = "HiearchyNet_pretrained"
GHCNN = "GHCNN"
GHCNN_PRE = "GHCNN_pretrained"

# sweep names

# architecture
# EFFICIENTNET_V2_S = "Efficient Net v2 s"

# optimizer
OPTI_ADAM = "adam"

# evaluation metrics
EVAL_METRIC_ACCURACY = "acc"
EVAL_METRIC_MSE = "MSE"
EVAL_METRIC_ALL = "all"

# checkpoint & early stopping
CHECKPOINT_DEFAULT_TOP_N = 1
EARLY_STOPPING_DEFAULT = np.Inf

### preprocessing
# image size
H256_W256 = (256, 256)
H224_W224 = (224, 224)


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

V10_ANNOTATED_MEAN = [0.42359381914138794, 0.4423922896385193, 0.4317706823348999]
V10_ANNOTATED_SD = [0.23015739023685455, 0.23571939766407013, 0.2612236738204956]

V10_ANNOTATED_MEAN = [0.42359381914138794, 0.4423922896385193, 0.4317706823348999]
V10_ANNOTATED_SD = [0.23015739023685455, 0.23571939766407013, 0.2612236738204956]

V11_ANNOTATED_MEAN = [0.423864483833313, 0.44273582100868225, 0.4323558807373047]
V11_ANNOTATED_SD = [0.23039115965366364, 0.23605570197105408, 0.26162344217300415]

V12_ANNOTATED_MEAN = [0.423864483833313, 0.44273582100868225, 0.4323558807373047]
V12_ANNOTATED_SD = [0.23039112985134125, 0.2360556572675705, 0.26162344217300415]



MODELSTRUCTURE = "use_model_structure"
GROUNDTRUTH = "use_ground_truth"
CC = "CC"

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
CLM = "clm"
CORN = 'corn'

V12_ANNOTATED_ASPHALT_MEAN = [0.4181784987449646, 0.44584745168685913, 0.4459141194820404]
V12_ANNOTATED_ASPHALT_SD = [0.23218706250190735, 0.2414448857307434, 0.27008959650993347]
