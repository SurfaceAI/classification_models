import sys
sys.path.append('.\\')
sys.path.append('..')

import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import os
from utils import preprocessing
from utils import helper
from utils import constants
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from utils import general_config

config = dict(
    project = "road-surface-classification-type",
    name = "VGG16",
    save_name = 'VGG16.pt',
    architecture = "VGG16",
    dataset = 'V4', #'annotated_images',
    label_type = 'predicted', #'annotated', #'predicted'
    seed = 42,
    image_size_h_w = (256, 256),
    normalization = 'imagenet', # None, # 'imagenet', 'from_data'
    crop = 'lower_middle_third',
    selected_classes = [constants.ASPHALT,
                        constants.CONCRETE,
                        constants.SETT,
                        constants.UNPAVED,
                        constants.PAVING_STONES,
    ]

)


### 1. Load testing data
torch.manual_seed(config.get('seed'))

general_transform = {
    'resize': config.get('image_size_h_w'),
    'crop': config.get('crop'),
    'normalize': config.get('normalization'),
}


test_images = preprocessing.create_test_dataset(config.get('dataset'),
                                            config.get('label_type'),
                                            general_transform,
                                            random_state=config.get('seed'))

# for image in test_images:
#     plt.imshow(image)
#     plt.axis('off')  # Turn off axis labels
#     plt.show()

### 2. Load the model
model_name = "VGG16.pt"
model = torch.load(os.path.join(general_config.save_path, model_name))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

predictions = []

import numpy as np
import torch

# Assuming test_images is a NumPy array containing images
with torch.no_grad():
    for index in range(len(test_images)):
        # Get a single item from the dataset
        image = test_images[index]
        image_file_name = test_images.total_imgs[index]
        image_id, _ = os.path.splitext(image_file_name)

        # Add one dimension for prediction
        image = image.unsqueeze(0)
        
        # Move the image tensor to the same device as the model
        image = image.to(device)
        
        # Forward pass to obtain predictions
        outputs = model(image)
        
        # Assuming your model outputs logits, apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class index
        predicted_class = torch.argmax(probabilities, dim=1).item()
        class_name = config["selected_classes"][predicted_class]
        class_probability = round(probabilities[0, predicted_class].item(), 2)

        # Append the prediction information as a tuple to the list
        predictions.append((image_id, class_name, class_probability))

model_predictions = pd.DataFrame(predictions, columns=['image_id', 'model_prediction', 'class_probability'])

# Read OSM tags table
file_path = r"C:\Users\esthe\Documents\GitHub\classification_models\data\v4_labels.csv"
OSM_tags = pd.read_csv(file_path, usecols=["id", "surface_osm"])

# Rename columns for consistency
OSM_tags.columns = ['image_id', 'osm_prediction']

# Convert 'image_id' columns to Int64
OSM_tags['image_id'] = OSM_tags['image_id'].astype('Int64')

# Assuming 'predictions' is your existing DataFrame
model_predictions['image_id'] = model_predictions['image_id'].astype('Int64')

OSM_tags
model_predictions

# Merge predictions and OSM_tags based on 'image_id'
merged_predictions = pd.merge(predictions, OSM_tags, on="image_id")
merged_predictions.to_csv(r"C:\Users\esthe\Documents\GitHub\classification_models\data\osm_model_predictions.csv", index=False)

# Accuracy and confusion matrix

accuracy = accuracy_score(merged_predictions["osm_prediction"], merged_predictions["model_prediction"])
accuracy_str = f'Accuracy: {accuracy * 100:.2f}%'
#29,18

cm = confusion_matrix(merged_predictions["osm_prediction"], merged_predictions["model_prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["selected_classes"])
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".0f")
plt.xticks(rotation=45, ha="right")

plt.xlabel('model prediction')
plt.ylabel('osm prediction')


plt.title('Confusion Matrix')
plt.show()

plt.savefig(r"C:\Users\esthe\Documents\GitHub\classification_models\data\confusion_matrix_plot.png")
