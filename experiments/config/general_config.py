from pathlib import Path

from src import constants

gpu_kernel = 1
wandb_mode = constants.WANDB_MODE_ON

ROOT_DIR = Path(__file__).parent.parent.parent
data_training_path = ROOT_DIR / "data" / "training"
trained_model_path = ROOT_DIR / "trained_models"
data_testing_path = ROOT_DIR / "data" / "testing"
data_path = ROOT_DIR / "data"

# selected_surface_classes = [constants.ASPHALT,
#                             constants.CONCRETE,
#                             constants.PAVING_STONES,
#                             constants.SETT,
#                             constants.UNPAVED,
# ]

# TODO: infinite depth: list of dict of list of dict // dict of dict of dict ...?
selected_classes = {
    constants.ASPHALT: [
        constants.EXCELLENT,
        constants.GOOD,
        constants.INTERMEDIATE,
        constants.BAD,
    ],
    constants.CONCRETE: [
        constants.EXCELLENT,
        constants.GOOD,
        constants.INTERMEDIATE,
        constants.BAD,
    ],
    constants.PAVING_STONES: [
        constants.EXCELLENT,
        constants.GOOD,
        constants.INTERMEDIATE,
        constants.BAD,
    ],
    constants.SETT: [constants.GOOD, constants.INTERMEDIATE, constants.BAD],
    constants.UNPAVED: [constants.INTERMEDIATE, constants.BAD, constants.VERY_BAD],
}

general_transform = dict(
    resize=constants.H256_W256,
    crop=constants.CROP_LOWER_MIDDLE_THIRD,
    normalize=constants.NORM_DATA,
)

augmentation = dict(
    random_horizontal_flip=True,
    random_rotation=10,
)

# dataset = constants.V4
# label_type = constants.ANNOTATED
dataset = "V6/annotated"

seed = 42

validation_size = 0.2

valid_batch_size = 48

checkpoint_top_n = 1

early_stop_thresh = 5

save_state = True
