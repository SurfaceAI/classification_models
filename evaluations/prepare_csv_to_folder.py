import sys

sys.path.append(".")

from src import constants as const
from experiments.config  import global_config

import pandas as pd
import os
import shutil

root_data_path = global_config.global_config.get('data_path')
metadata = "road_scenery_experiment/metadata"
version_file_name = "annotations_scenery_v5.csv"

v1_dataset = "V1_0/s_1024"
v12_no_street = "training/V12/annotated/no_street"
v12_not_rec = "training/V12/annotated/not_recognizable"

image_folders = {
    "v1": v1_dataset,
    "v12_no_street": v12_no_street,
    "v12_not_rec": v12_not_rec,
}

version_folder = "road_scenery_experiment/v5"

version_df = pd.read_csv(os.path.join(root_data_path, metadata, version_file_name))

def find_original_folder(image_id):
    image_filename = f"{image_id}.jpg"
    for f, folder in image_folders.items():
        image_path = os.path.join(root_data_path, folder, image_filename)
        if os.path.exists(image_path):
            return f
    return ""  # not found

# Bilder in die entsprechenden Folder verschieben
def find_image_path(image_id):
    image_filename = f"{image_id}.jpg"
    for folder in image_folders.values():
        image_path = os.path.join(root_data_path, folder, image_filename)
        if os.path.exists(image_path):
            return image_path
    return None

for _, row in version_df.iterrows():
    image_id, folder_structure = row["image_id"], row["road_scenery"]
    image_path = find_image_path(image_id)
    
    if image_path:
        target_folder = os.path.join(root_data_path, version_folder, *folder_structure.split("__"))
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(image_path, os.path.join(target_folder, f"{image_id}.jpg"))
    else:
        print(f"Image id {image_id} not found in folders.")