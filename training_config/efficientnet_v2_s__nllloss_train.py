import sys
sys.path.append('.')

from torch import nn, optim
from model_config import efficientnet_models
from utils import training, constants, general_config
import os


# config
config = dict(
    project = "road-surface-classification-type", # general? (different for sweep and train, or all sweep another?) # main input argument?
    name = "efficient net",                       # model? # main input argument?
    save_name = 'efficientnet_v2_s__nllloss.pt',  # model?
    architecture = "Efficient Net v2 s",    # model?
    dataset = 'V4',                         # general? +track
    label_type = 'annotated',               # general? +track
    batch_size = 48,         # train?
    valid_batch_size = 48,   # general? no track?
    epochs = 2,              # train?
    learning_rate = 0.0003,  # train?
    seed = 42,               # general? +track
    validation_size = 0.2,   # general? +track
    image_size_h_w = (256, 256),   # general, model or train?
    crop = 'lower_middle_third',   # general_config?
    normalization = 'from_data', # None, # 'imagenet', 'from_data'  # general_config?
    # norm_mean = [0.485, 0.456, 0.406],
    # norm_std = [0.229, 0.224, 0.225],
    selected_classes = [constants.ASPHALT,   # general_config?
                        constants.CONCRETE,
                        constants.SETT,
                        constants.UNPAVED,
                        constants.PAVING_STONES,
    ]

)

augmentation = dict(   # general_config, YES/NO only in train_config/sweep?
    random_horizontal_flip = True,
    random_rotation = 10,
)

# TODO: optimize only last layer? how to define parameters to optimize?
optimizer = optim.Adam   # train/sweep?

# loss, reduction='sum'
criterion = nn.NLLLoss()   # model_config? no track?

os.environ["WANDB_MODE"] = general_config.wandb_mode

# train model
training.config_and_train_model(config, efficientnet_models.CustomEfficientNetV2SLogsoftmax, optimizer, criterion, augmentation)