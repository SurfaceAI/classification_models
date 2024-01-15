from collections import OrderedDict

from torch import nn
from torchvision import models

# TODO: wie kann ich aus dem modell eine Klasse machen, die die Modell attribute und methoden besitzt,
# aber die Größe des letzten layers definiert werden kann bei init?


def load_model(num_classes):

    # model
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

    # # Unfreeze parameters
    # for param in model.parameters():
    #     param.requires_grad = True

    # adapt output layer
    fc = nn.Sequential(
        OrderedDict(
            [("fc1", nn.Linear(1280, num_classes)), ("output", nn.LogSoftmax(dim=1))]
        )
    )

    model.classifier[1] = fc

    optimizer_layers = [model.classifier[1]]

    return model, optimizer_layers
