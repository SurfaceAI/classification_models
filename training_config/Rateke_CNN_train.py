import sys
sys.path.append('.')
sys.path.append('..')


import torch
from torch import nn, optim
from model_config import Rateke_CNN_model
from utils import training_esther

# import sys
# sys.path.append(r'C:\Users\esthe\Documents\GitHub\classification_models')
# import training


# config
config = dict(
    project = "road-surface-classification-type",
    name = "Rateke CNN",
    save_name = 'Simple_CNN.pt',
    architecture = "Simple CNN not pretrained",
    dataset = 'V4', #'annotated_images',
    label_type = 'annotated', #'predicted
    batch_size = 16,
    valid_batch_size = 48,
    epochs = 2,
    learning_rate = 0.0002,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (768, 768),
    crop_size = [512, 256, 256, 256],
    norm_mean = [0.485, 0.456, 0.406],
    norm_std = [0.229, 0.224, 0.225],

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
training_esther.config_and_train_model(config, Rateke_CNN_model.load_model, optimizer, criterion, augmentation)
