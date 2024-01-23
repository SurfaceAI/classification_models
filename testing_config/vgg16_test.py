import sys
sys.path.extend([".\\", ".."])
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt
from utils import preprocessing, constants, general_config


# Configuration
config = {
    "project": "road-surface-classification-type",
    "name": "VGG16",
    "save_name": "VGG16.pt",
    "architecture": "VGG16",
    "dataset": "V4",
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
model_name = "VGG16.pt"
model = torch.load(os.path.join(general_config.save_path, model_name))
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
        class_probability = round(probabilities[0, predicted_class].item(), 2)

        # Append the prediction information as a tuple to the list
        predictions.append((image_id, class_name, class_probability))

# Create a DataFrame with model predictions
model_predictions = pd.DataFrame(
    predictions, columns=["image_id", "model_prediction", "class_probability"]
)
model_predictions["image_id"] = model_predictions["image_id"].astype("Int64")

# Read OSM tags table
file_path = r"C:\Users\esthe\Documents\GitHub\classification_models\data\v4_labels.csv"
OSM_tags = pd.read_csv(file_path, usecols=["id", "surface_osm"])
OSM_tags.columns = ["image_id", "osm_prediction"]
OSM_tags["image_id"] = OSM_tags["image_id"].astype("Int64")

# Merge predictions and OSM_tags based on 'image_id'
merged_predictions = pd.merge(model_predictions, OSM_tags, on="image_id")
merged_predictions.to_csv(
    r"C:\Users\esthe\Documents\GitHub\classification_models\data\osm_model_predictions_1.csv",
    index=False,
)

# Accuracy and confusion matrix
accuracy = accuracy_score(
    merged_predictions["osm_prediction"], merged_predictions["model_prediction"]
)
accuracy_str = f"Accuracy: {accuracy * 100:.2f}%"
print(accuracy_str)

# Confusion matrix
cm = confusion_matrix(
    merged_predictions["osm_prediction"], merged_predictions["model_prediction"]
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=config["selected_classes"]
)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".0f")
plt.xticks(rotation=45, ha="right")
plt.xlabel("model prediction")
plt.ylabel("osm prediction")
plt.title("Confusion Matrix")
plt.show()

# Save confusion matrix plot
plt.savefig(
    r"C:\Users\esthe\Documents\GitHub\classification_models\data\confusion_matrix_plot.png"
)

# Classification report
classification_rep = classification_report(
    merged_predictions["osm_prediction"],
    merged_predictions["model_prediction"],
    labels=config["selected_classes"],
)
print(classification_rep)
