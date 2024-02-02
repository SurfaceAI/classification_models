import sys
sys.path.append('.')
sys.path.append('..')


import torch
from torch import nn, optim
from model_config import Rateke_CNN_model
from utils import training, constants

# config
config = dict(
    project = "road-surface-classification-combined",
    name = "Rateke_CNN_flat",
    save_name = 'Simple_CNN_flat',
    architecture = "Simple_CNN_not_pretrained",
    dataset = 'V4', 
    label_type = 'annotated', #'predicted
    batch_size = 32,
    valid_batch_size = 32,
    epochs = 20,
    learning_rate = 0.0002,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (256, 256),
    crop = 'lower_middle_third',
    normalization = 'imagenet', # None, # 'imagenet', 'from_data'
    selected_classes=[
            constants.ASPHALT,
            constants.CONCRETE,
            constants.PAVING_STONES,
            constants.SETT,
            constants.UNPAVED,
        ],
    selected_quality_classes = {
        constants.ASPHALT: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
        constants.CONCRETE: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
        constants.PAVING_STONES: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
        constants.SETT: [constants.GOOD, constants.INTERMEDIATE, constants.BAD],
        constants.UNPAVED: [constants.INTERMEDIATE, constants.BAD, constants.VERY_BAD],
    }

)

augmentation = dict(
    random_horizontal_flip = True,
    random_rotation = 10,
)

# TODO: optimize only last layer? how to define parameters to optimize?
optimizer = optim.Adam

# loss, reduction='sum'
criterion = nn.CrossEntropyLoss()


# die Funktion create_flat_train_validation_datasets muss in training.py durch create_flat_train_validation_dataset, damit die Klassen 'flat'
# eingelesen werden. 

# train model
training.config_and_train_model(config, Rateke_CNN_model.ConvNet, optimizer, criterion, augmentation=augmentation)

