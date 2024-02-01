import sys
sys.path.append('.')
sys.path.append('..')


import torch
from torch import nn, optim
from src.architecture import Rateke_CNN
from src.models import training
from src import constants

# import sys
# sys.path.append(r'C:\Users\esthe\Documents\GitHub\classification_models')
# import training


# config
config = dict(
    project = "road-surface-classification-type",
    name = "Rateke CNN",
    save_name = 'Simple_CNN_type',
    architecture = "Simple CNN not pretrained",
    dataset = 'V4', #'annotated_images',
    #dataset_class = 'FlattenFolders', #'FlattenFolders', #'PartialImageFolder'
    label_type = 'annotated', #'predicted
    batch_size = 16,
    valid_batch_size = 48,
    epochs = 2,
    learning_rate = 0.0002,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (256, 256),
    crop = 'lower_middle_third',
    normalization = 'imagenet', # None, # 'imagenet', 'from_data'
    # norm_mean = [0.485, 0.456, 0.406],
    # norm_std = [0.229, 0.224, 0.225],
    selected_classes=[
            constants.ASPHALT,
            constants.CONCRETE,
            constants.PAVING_STONES,
            constants.SETT,
            constants.UNPAVED,
        ]
)

augmentation = dict(
    random_horizontal_flip = True,
    random_rotation = 10,
)

# TODO: optimize only last layer? how to define parameters to optimize?
optimizer = optim.Adam

# loss, reduction='sum'
criterion = nn.CrossEntropyLoss()

# TODO: kann ich eine Klasse übergeben (soll erst in zielfunlktion initialisiert werden mit num_classes)?


# train model
training.config_and_train_model(config, Rateke_CNN.ConvNet, optimizer, criterion, augmentation=augmentation)

