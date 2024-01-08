import sys
sys.path.append('.')
sys.path.append('..')

from torch import nn, optim
from model_config import vgg16_model
from utils import training_esther

# config
config = dict(
    project = "road-surface-classification-type",
    name = "VGG16",
    save_name = 'VGG16.pt',
    architecture = "VGG16",
    dataset = 'annotated_images',
    batch_size = 32,
    valid_batch_size = 32,
    epochs = 3,
    learning_rate = 0.0001,
    seed = 42,
    validation_size = 0.2,
    image_size_h_w = (768, 768), #if cropped, three times the final image size, if not cropped the final image size
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
training_esther.config_and_train_model(config, vgg16_model.load_model, optimizer, criterion, augmentation)
