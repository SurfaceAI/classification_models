### hint: last part should be 'normalization values', this is extended by calculation of new dataset normalization

# weights and biases
WANDB_MODE_ON = 'online'
WANDB_MODE_OFF = 'offline'
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

# classification level
SURFACE = "surface"
SMOOTHNESS = "smoothness"

# dataset
V0 = 'V0'
V1 = 'V1'
V2 = 'V2'
V3 = 'V3'
V4 = 'V4'

# label type
ANNOTATED = 'annotated'

# project names
PROJECT_SURFACE_FIXED = "road-surface-classification-type"
PROJECT_SURFACE_SWEEP = "sweep-road-surface-classification-type"

# model names
EFFICIENTNET = "efficient_net_v2_s__logsoftmax"

# sweep names

# architecture
EFFICIENTNET_V2_S = "Efficient Net v2 s"

# optimizer
OPTI_ADAM = 'adam'

### preprocessing
# image size
H256_W256 = (256, 256)

# crop
CROP_LOWER_MIDDLE_THIRD = 'lower_middle_third'

# augmentation
AUGMENT_TRUE = 'yes'
AUGMENT_FALSE = 'no'

# normalization
NORM_IMAGENET = 'imagenet'
NORM_DATA = 'from_data'

# normalization values
IMAGNET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_SD = [0.229, 0.224, 0.225]

V4_ANNOTATED_MEAN = [0.4205051362514496, 0.439302921295166, 0.42983368039131165]
V4_ANNOTATED_SD = [0.23013851046562195, 0.23709532618522644, 0.2626153826713562]
