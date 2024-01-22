from pathlib import Path
from utils import constants
gpu_kernel = 1
wandb_mode = constants.WANDB_MODE_ON

ROOT_DIR = Path(__file__).parent.parent
training_data_path = ROOT_DIR / 'data' / 'training'
save_path = ROOT_DIR / 'models'

selected_classes = [constants.ASPHALT,
                    constants.CONCRETE,
                    constants.SETT,
                    constants.UNPAVED,
                    constants.PAVING_STONES,
]

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

valid_batch_size = 48


