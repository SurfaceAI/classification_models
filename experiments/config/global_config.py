from pathlib import Path

from src import constants

ROOT_DIR = Path(__file__).parent.parent.parent

global_config = dict(
    gpu_kernel = 0,
    wandb_mode = constants.WANDB_MODE_ON,
    wandb_on=False,
    root_data = str(ROOT_DIR / "data" / "training"),
    root_model = str(ROOT_DIR / "trained_models"),
    root_predict = str(ROOT_DIR / "data" / "training" / "prediction"),
    data_testing_path = str(ROOT_DIR / "data" / "testing"),
    data_path = str(ROOT_DIR / "data"),
    evaluation_path = str(ROOT_DIR / "evaluations"),
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
    },
    transform = dict(
        resize=constants.H256_W256,
        crop=constants.CROP_LOWER_MIDDLE_THIRD,
        normalize=constants.NORM_DATA,
    ),
    augment = dict(
        random_horizontal_flip=True,
        random_rotation=10,
    ),
    dataset = "V12/annotated",
    seed = 42,
    validation_size = 0.2,
    valid_batch_size = 128,
    checkpoint_top_n = 1,
    early_stop_thresh = constants.EARLY_STOPPING_DEFAULT,
    save_state = False,
)   