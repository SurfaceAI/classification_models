import torch
from torch import nn, optim

from src import constants as const
from src.architecture import Rateke_CNN, efficientnet, vgg16


def model_name_to_config(model_name):
    model_cfg = {}
    if model_name == "efficientNetV2SLogsoftmax":
        model_cfg["architecture"] = efficientnet.architecture
        model_cfg["criterion"] = nn.NLLLoss()
        model_cfg["model_cls"] = efficientnet.CustomEfficientNetV2SLogsoftmax
        model_cfg["logits_to_prob"] = torch.exp

    if model_name == "rateke":
        model_cfg["architecture"] = Rateke_CNN.architecture
        model_cfg["criterion"] = nn.CrossEntropyLoss()
        model_cfg["model_cls"] = Rateke_CNN.ConvNet
        model_cfg["logits_to_prob"] = lambda x: nn.functional.softmax(x, dim=1)

    if model_name == "vgg16":
        model_cfg["architecture"] = vgg16.architecture
        model_cfg["criterion"] = nn.CrossEntropyLoss()
        model_cfg["model_cls"] = vgg16.CustomVGG16
        model_cfg["logits_to_prob"] = lambda x: nn.functional.softmax(x, dim=1)

    if model_name == "vgg16Regression":
        model_cfg["architecture"] = vgg16.architecture
        model_cfg["criterion"] = nn.MSELoss()
        model_cfg["model_cls"] = vgg16.CustomVGG16

    return model_cfg


def optim_name_to_class(optimizer_name):
    if optimizer_name == const.OPTI_ADAM:
        return optim.Adam
