import sys

sys.path.append("./")

import efficientnet_v2_s_logsoftmax_model
from torch import nn, optim

import src.training as training

# config
config = dict(
    project="road-surface-classification-type",
    name="efficient net",
    save_name="efficientnet_v2_s__nllloss.pt",
    architecture="Efficient Net v2 s",
    dataset="V0",
    batch_size=48,
    valid_batch_size=48,
    epochs=2,
    learning_rate=0.0003,
    seed=42,
    validation_size=0.2,
    image_size_h_w=(256, 256),
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
)

augmentation = dict(
    random_horizontal_flip=True,
    random_rotation=10,
)

# TODO: optimize only last layer? how to define parameters to optimize?
optimizer = optim.Adam

# loss, reduction='sum'
criterion = nn.NLLLoss()

# TODO: kann ich eine Klasse übergeben (soll erst in zielfunlktion initialisiert werden mit num_classes)?


# train model
training.config_and_train_model(
    config,
    efficientnet_v2_s_logsoftmax_model.load_model,
    optimizer,
    criterion,
    augmentation,
)
