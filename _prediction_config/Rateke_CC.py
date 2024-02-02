import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os
import pandas as pd
from utils import preprocessing, constants, general_config
from model_config import Rateke_CNN_model
from torch.utils.data import Subset

#before running this script, you have to have all images in the data folder, e.g. "V4/predicted"

# Configuration
config = {
    "project": "road-surface-classification-CC",
    "type_model": "Simple_CNN_type", #"VGG16_type"
    "quality_models": "Simple_CNN_quality", #"VGG16_quality"
    #"name": "Rateke",
    #"save_name": "Simple_CNN.pt",
    #"architecture": "Rateke",
    "dataset": "V4",
    "label_type": "predicted", #"predicted" #"annotated"
    "seed": 42,
    "image_size_h_w": (256, 256),
    "normalization": "imagenet",
    "crop": "lower_middle_third",
    "selected_classes": [
        constants.ASPHALT,
        constants.CONCRETE,
        constants.PAVING_STONES,
        constants.SETT,
        constants.UNPAVED,
    ],
    "selected_quality_classes" : {
         constants.ASPHALT: [constants.BAD, constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE],
        constants.CONCRETE: [constants.BAD, constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE],
        constants.PAVING_STONES: [constants.BAD, constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE],
        constants.SETT: [constants.BAD, constants.GOOD, constants.INTERMEDIATE],
        constants.UNPAVED: [constants.BAD, constants.INTERMEDIATE, constants.VERY_BAD],
    }

}

# Set seed for reproducibility
torch.manual_seed(config["seed"])

# Load testing data
general_transform = {
    "resize": config["image_size_h_w"],
    "crop": config["crop"],
    "normalize": config["normalization"],
}
test_images = preprocessing.create_test_dataset(
    config["dataset"], config["label_type"], general_transform, random_state=config["seed"]
)


# Load all models
quality_model_names = [f"{config['quality_models']}_{type_class}" for type_class in config["selected_classes"]]

#load type model
model = Rateke_CNN_model.ConvNet(len(config["selected_classes"])) 
model.load_state_dict(torch.load(os.path.join(general_config.save_path, config["type_model"])))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#type predictions
predictions = []

with torch.no_grad():
    for index, image in enumerate(test_images):
        image_file_name = test_images.total_imgs[index]
        #image_id, _ = os.path.splitext(image_file_name) #entfernt einfach nur das .jpg bei den image files

        # Add one dimension for prediction
        image = image.unsqueeze(0).to(device)

        # Forward pass to obtain predictions
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the predicted class index
        predicted_class = torch.argmax(probabilities, dim=1).item()
        class_name = config["selected_classes"][predicted_class]
        class_probability = round(probabilities[0, predicted_class].item(), 3)

        # Append the prediction information as a tuple to the list
        predictions.append((image_file_name, class_name, class_probability))
        
        

# Create a DataFrame with model predictions for type so we can easily loop over the different types
model_predictions = pd.DataFrame(
    predictions, columns=["image_file_name", "predicted_class", "class_probability"]
)


# Loop over all type_classes
combined_predictions = []

for i, type_class in enumerate(config["selected_classes"]):
    class_images = model_predictions[model_predictions['predicted_class'] == type_class]

    matched_indices = [idx for idx, image_filename in enumerate(test_images.total_imgs) if image_filename in class_images["image_file_name"].tolist()]
    # Create a Subset using the obtained indices
    type_prediction_images = Subset(test_images, matched_indices)

    model = Rateke_CNN_model.ConvNet(len(config["selected_quality_classes"][type_class]))  # TODO: Rewrite the function load.model to class
    model.load_state_dict(torch.load(os.path.join(general_config.save_path, quality_model_names[i])))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for index, image in enumerate(type_prediction_images):
            image_file_name = test_images.total_imgs[index]  # Uncomment this line
            # image_id, _ = os.path.splitext(image_file_name)

            # Add one dimension for prediction
            image = image.unsqueeze(0).to(device)

            # Forward pass to obtain predictions
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get the predicted class index
            predicted_class = torch.argmax(probabilities, dim=1).item()
            class_name = config["selected_quality_classes"][type_class][predicted_class]
            class_probability = round(probabilities[0, predicted_class].item(), 3)

            # Append the prediction information as a tuple to the list
            combined_predictions.append((image_file_name, type_class, class_name, class_probability))
            combined_predictions_df = pd.DataFrame(combined_predictions, columns=['image_file_name', 'type_class', 'quality_class', 'score'])

class_distribution = combined_predictions_df.groupby('type_class')['quality_class'].value_counts()

print(class_distribution)