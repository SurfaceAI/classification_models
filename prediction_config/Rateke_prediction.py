import sys
sys.path.append('.')
sys.path.append('..')
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocessing, constants, general_config
from model_config import Rateke_CNN_model


# Configuration
config = {
    "project": "road-surface-classification-type",
    "name": "Rateke_CNN",
    "save_name": "Simple_CNN_type",
    "architecture": "Simple_CNN_not_pretrained",
    "dataset": "V5_c1",
    "label_type": "predicted",
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
    "selected_quality_classes": {
                constants.ASPHALT: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.CONCRETE: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.PAVING_STONES: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.SETT: [constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.UNPAVED: [constants.INTERMEDIATE, constants.BAD, constants.VERY_BAD],
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


# Load the model
model = Rateke_CNN_model.ConvNet(len(config["selected_classes"]))
model.load_state_dict(torch.load(os.path.join(general_config.save_path, config["save_name"])))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Make predictions
predictions = []

with torch.no_grad():
    for index, image in enumerate(test_images):
        image_file_name = test_images.total_imgs[index]
        image_id, _ = os.path.splitext(image_file_name)

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
        predictions.append((image_id, class_name, class_probability))

# Create a DataFrame with model predictions
#Todo: not sure if this is necessary or if we just stick to the list of predictions
model_predictions = pd.DataFrame(
    predictions, columns=["image_id", "model_prediction", "class_probability"]
)
model_predictions["image_id"] = model_predictions["image_id"].astype("Int64")

#save to csv
model_predictions.to_csv(
    fr"{general_config.data_path}\model_predictions_{config['dataset']}_{config['label_type']}.csv",
    index=False,
)
