from experiments.training_config import rateke_cnn_default, vgg16_default
from src import train_model_esther


def get_config(config_name):
    # call general config
    config = general_config.config

    # call model config
    config_model = rateke_cnn_default.config

    # append and replace by config_model

    return config


# config = vgg16_default.config
config.learning_rate = 0.0001

train_model_esther.config_and_train_model(get_config("rakete_cnn"))
