from src.architecture import efficientnet
from src import constants
from torch import nn, optim

def model_name_to_config(model_name):
    model_cfg = {}
    if model_name == 'efficientNetV2SLogsoftmax':
        model_cfg["architecture"] = efficientnet.architecture
        model_cfg["criterion"] = nn.NLLLoss()
        model_cfg["model_cls"] = efficientnet.CustomEfficientNetV2SLogsoftmax

    return model_cfg

def optim_name_to_class(optimizer_name):

    if optimizer_name == constants.OPTI_ADAM:

        return optim.Adam
    
