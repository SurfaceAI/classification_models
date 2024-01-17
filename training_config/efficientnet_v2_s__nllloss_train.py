import sys
sys.path.append('.')

from torch import nn, optim
from model_config import efficientnet_models
from utils import training, constants


# config
config = dict(
    project = "road-surface-classification-type",
    name = "efficient net",
    save_name = 'efficientnet_v2_s__nllloss.pt',
    architecture = "Efficient Net v2 s",
    dataset = 'V4', #'annotated_images',
    label_type = 'annotated', #'predicted
    batch_size = 48,
    valid_batch_size = 48,
    epochs = 2,
    learning_rate = 0.0003,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (256, 256),
    crop = 'lower_middle_third',
    normalization = 'from_data', # None, # 'imagenet', 'from_data'
    # norm_mean = [0.485, 0.456, 0.406],
    # norm_std = [0.229, 0.224, 0.225],
    selected_classes = [constants.ASPHALT,
                        constants.CONCRETE,
                        constants.SETT,
                        constants.UNPAVED,
                        constants.PAVING_STONES,
    ]

)

augmentation = dict(
    random_horizontal_flip = True,
    random_rotation = 10,
)

# TODO: optimize only last layer? how to define parameters to optimize?
optimizer = optim.Adam

# loss, reduction='sum'
criterion = nn.NLLLoss()

# TODO: kann ich eine Klasse übergeben (soll erst in zielfunlktion initialisiert werden mit num_classes)?


# train model
training.config_and_train_model(config, efficientnet_models.CustomEfficientNetV2SLogsoftmax, optimizer, criterion, augmentation)