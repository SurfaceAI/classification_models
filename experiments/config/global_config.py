from pathlib import Path

from src import constants

ROOT_DIR = Path(__file__).parent.parent.parent

global_config = dict(
    gpu_kernel = 0,
    wandb_mode = constants.WANDB_MODE_ON, # TODO
    wandb_on=True,# TODO
    # root_data = str(ROOT_DIR / "data" / "training"),
    root_data = str(ROOT_DIR / "data"),
    root_model = str(ROOT_DIR / "trained_models"),
    # root_predict = str(ROOT_DIR / "data" / "training" / "prediction"),
    root_predict = str(ROOT_DIR / "prediction"),
    # data_testing_path = str(ROOT_DIR / "data" / "testing"),
    # data_path = str(ROOT_DIR / "data"),
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
        resize=constants.H384_W384,
        crop=constants.CROP_LOWER_MIDDLE_HALF,
        normalize=constants.NORM_DATA,
    ),
    augment = dict(
        random_horizontal_flip=True,
        random_rotation=10,
    ),
    # dataset = "V12/annotated", # TODO
    seed = 42,
    validation_size = 0.2,
    # valid_batch_size = 48,
    batch_size = 16,
    checkpoint_top_n = constants.CHECKPOINT_DEFAULT_TOP_N,
    early_stop_thresh = 5,
    save_state = False,
)