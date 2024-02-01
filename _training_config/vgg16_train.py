import sys
sys.path.append('.')
sys.path.append('..')

from torch import nn, optim
from src.architecture import vgg16
from src.models import training
from src import constants

config = dict(
    project = "road-surface-classification-type",
    name = "VGG16",
    save_name = 'VGG16.pt',
    architecture = "VGG16",
    dataset = 'V4', #'annotated_images',
    label_type = 'annotated', #'predicted
    #dataset_class = 'PartialImageFolder', #'FlattenFolders', #'PartialImageFolder'
    batch_size = 32,
    valid_batch_size = 32,
    epochs = 2,
    learning_rate = 0.0001,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (256, 256),
    crop = 'lower_middle_third',
    normalization = 'from_data', # None, # 'imagenet', 'from_data'
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
training.config_and_train_model(config, vgg16.CustomVGG16, optimizer, criterion, augmentation=augmentation)
