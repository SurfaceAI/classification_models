from model_config import efficientnet_models
from utils import constants
from torch import nn, optim

def config(model_name):
    model_cfg = {}
    if model_name == 'efficientNetV2SLogsoftmax':
        model_cfg["architecture"] = efficientnet_models.architecture
        model_cfg["criterion_cls"] = nn.NLLLoss
        model_cfg["model_cls"] = efficientnet_models.CustomEfficientNetV2SLogsoftmax

    return model_cfg

def optimizer_config(optimizer_name):

    if optimizer_name == constants.OPTI_ADAM:

        return optim.Adam