from torch import nn
from torchvision import models
from collections import OrderedDict


# TODO: wie kann ich aus dem modell eine Klasse machen, die die Modell attribute und methoden besitzt,
# aber die Größe des letzten layers definiert werden kann bei init?

def load_model(num_classes):

    # model
    model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    
    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False
    
    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features #number of output features in our last layer
    features = list(model.classifier.children())[:-1] #select features in our last layer
    features.extend([nn.Linear(num_features, num_classes)]) #add layer with output size num_classes
    
    model.classifier = nn.Sequential(*features) # Replace the model classifier

    # optimizer_layers = [model.classifier[-1]]

    return model

