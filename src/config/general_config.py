from pathlib import Path
from src import constants
gpu_kernel = 1
wandb_mode = constants.WANDB_MODE_ON

ROOT_DIR = Path(__file__).parent.parent.parent
training_data_path = ROOT_DIR / 'data' / 'training'
save_path = ROOT_DIR / 'trained_models'
test_data_path = ROOT_DIR / 'data' / 'testing'
data_path = ROOT_DIR / 'data'

# selected_surface_classes = [constants.ASPHALT,
#                             constants.CONCRETE,
#                             constants.PAVING_STONES,
#                             constants.SETT,
#                             constants.UNPAVED,
# ]

selected_classes = {constants.ASPHALT: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                    constants.CONCRETE: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                    constants.PAVING_STONES: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                    constants.SETT: [constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                    constants.UNPAVED: [constants.INTERMEDIATE, constants.BAD, constants.VERY_BAD],
}

general_transform = dict(
    resize = constants.H256_W256,
    crop = constants.CROP_LOWER_MIDDLE_THIRD,
    normalize = constants.NORM_DATA,
)

dataset = constants.V4
label_type = constants.ANNOTATED

seed = 42

validation_size = 0.2
    
augmentation = dict(
    random_horizontal_flip = True,
    random_rotation = 10,
)

valid_batch_size = 8 # 48

checkpoint_top_n = 2

early_stop_thresh = 2


