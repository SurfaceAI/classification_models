import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

file_path = r"C:\Users\esthe\Documents\GitHub\classification_models\data\v4_labels.csv"


#Read OSM tags from csv
OSM_tags = pd.read_csv(file_path, usecols=["id", "surface_osm"])
OSM_tags.columns = ["image_id", "osm_prediction"]
OSM_tags["image_id"] = OSM_tags["image_id"].astype("Int64")

#Merge predictions from pandas dataframe and OSM_tags based on 'image_id'
merged_predictions = pd.merge(model_predictions, OSM_tags, on="image_id")


#load last training session's validation data
valid_data = torch.load(os.path.join(general_config.save_path, "valid_data.pt")) #not sure if and how this is saved in the new setup
#this just strips the'.jpg' from the images file names
valid_data_numbers = [int(os.path.splitext(os.path.basename(path))[0]) for path, _ in valid_data.imgs]
#adding dummy variable that indicates whether a picture was a validation image (1 valid_image, 0 train_image)
merged_predictions["validation_data"] = merged_predictions["image_id"].isin(valid_data_numbers).astype(int)


#writing csv
model_predictions.to_csv(
    fr"{general_config.data_path}\model_predictions_{config['dataset']}_{config['label_type']}.csv",
    index=False,
)


#Accuracy
accuracy = accuracy_score(
    merged_predictions["osm_prediction"], merged_predictions["model_prediction"]
)
accuracy_str = f"Accuracy: {accuracy * 100:.2f}%"
print(accuracy_str)



#Confusion matrix
cm = confusion_matrix(
    merged_predictions["osm_prediction"], merged_predictions["model_prediction"]
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=config["selected_classes"]
)

# Plotting
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".0f")

# Adjusting axis names
plt.xticks(rotation=45, ha="right")
plt.xlabel("Model Prediction")
plt.ylabel("OSM Prediction")
plt.title("Confusion Matrix")

# Save confusion matrix plot
plt.savefig(
    fr"C:\Users\esthe\Documents\GitHub\classification_models\data\confusion_matrix_{config['dataset']}_{config['label_type']}.png"
)




#Alternatively: Classification report
classification_rep = classification_report(
    merged_predictions["osm_prediction"],
    merged_predictions["model_prediction"],
    labels=config["selected_classes"],
)
print(classification_rep)

# Save classification report to a text file
with open(r"C:\Users\esthe\Documents\GitHub\classification_models\data\classification_report_server.txt", "w") as f:
    f.write(classification_rep)