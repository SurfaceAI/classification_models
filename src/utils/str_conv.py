from src.architecture import efficientnet, Rateke_CNN, vgg16
from src import constants
from torch import nn, optim

def model_name_to_config(model_name):
    model_cfg = {}
    if model_name == 'efficientNetV2SLogsoftmax':
        model_cfg["architecture"] = efficientnet.architecture
        model_cfg["criterion"] = nn.NLLLoss()
        model_cfg["model_cls"] = efficientnet.CustomEfficientNetV2SLogsoftmax

    if model_name == 'rateke':
        model_cfg["architecture"] = Rateke_CNN.architecture
        model_cfg["criterion"] = nn.CrossEntropyLoss()
        model_cfg["model_cls"] = Rateke_CNN.ConvNet

    if model_name == 'vgg16':
        model_cfg["architecture"] = vgg16.architecture
        model_cfg["criterion"] = nn.CrossEntropyLoss()
        model_cfg["model_cls"] = vgg16.CustomVGG16

    return model_cfg

def optim_name_to_class(optimizer_name):

    if optimizer_name == constants.OPTI_ADAM:

        return optim.Adam
    
