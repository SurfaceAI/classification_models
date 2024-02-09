import sys

sys.path.append(".")

from experiments.config import config_helper, train_config
from src import constants
from src.models import training

project = constants.PROJECT_SMOOTHNESS_FIXED

name = "VGG16Regression"

level = constants.SMOOTHNESS

model = "vgg16Regression"  # TODO: constants.VGG16 ?

individual_params = train_config.fixed_params

# overwrite custom values
individual_params["eval_metric"] = constants.EVAL_METRIC_MSE
individual_params["epochs"] = 5
individual_params["is_regression"] = True

config = config_helper.fixed_config(individual_params=individual_params, model=model)

training.run_fixed_training(
    config=config,
    project=project,
    name=name,
    level=level,
    wandb_mode=constants.WANDB_MODE_OFF,
)
