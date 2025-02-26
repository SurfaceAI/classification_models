import sys

sys.path.append(".")

from src import constants as const
from experiments.config  import global_config

import pandas as pd
import os
from torchvision import datasets
from PIL import Image

root_data_path = global_config.global_config.get('data_path')
# # dataset = "road_scenery_experiment/classified_images_add_on_c"
# dataset_list = [
#     "road_scenery_experiment/classified_images",
#     "road_scenery_experiment/classified_images_add_on",
#     "road_scenery_experiment/classified_images_add_on_b",
#     "road_scenery_experiment/classified_images_add_on_c",
# ]
metadata = "road_scenery_experiment/metadata"
corr_file_name = 'annotations_corr.csv'
corr_file_name_id_corr = 'annotations_corr_id.csv'
corr_file_name_update = 'annotations_corr_update.csv'
v4_file_name = "annotations_scenery_v4.csv"
v5_file_name = "annotations_scenery_v5.csv"

v1_dataset = "V1_0/s_1024"
v12_no_street = "training/V12/annotated/no_street"
v12_not_rec = "training/V12/annotated/not_recognizable"

image_folders = {
    "v1": v1_dataset,
    "v12_no_street": v12_no_street,
    "v12_not_rec": v12_not_rec,
}

folder_map = {
    "1_1_road_general": "1_1_road__1_1_road_general",
    "1_2_cycleway": "1_2_bicycle__1_2_cycleway",
    "1_2_lane": "1_2_bicycle__1_2_lane",
    "1_3_footway": "1_3_pedestrian__1_3_footway",
    "1_4_path_unspecified": "1_4_path__1_4_path_unspecified",
    "2_1_all": "2_1_no_focus_no_street__2_1_all",
}


# new_folder = "road_scenery_experiment/corrections"


# # CSV-Datei einlesen
# df = pd.read_csv(os.path.join(root_data_path, metadata, corr_file_name_id_corr))

# def find_original_folder(image_id):
#     image_filename = f"{image_id}.jpg"
#     for f, folder in image_folders.items():
#         image_path = os.path.join(root_data_path, folder, image_filename)
#         if os.path.exists(image_path):
#             return f
#     return ""  # not found

# # Original-Folder ermitteln
# df["original_folder"] = df["image_id"].apply(find_original_folder)

# # Aktualisierte CSV speichern
# df.to_csv(os.path.join(root_data_path, metadata, corr_file_name_update), index=False)


# # CSV-Datei korrigieren

# def find_corrected_id(image_id):
#     original_folder = find_original_folder(image_id)
#     if original_folder == "":
    
#         possible_ids = [str(image_id)[:-1] + str(i) for i in range(10)]  # Erzeuge 10 mögliche Varianten
#         found_images = []

#         for folder in image_folders.values():
#             for possible_id in possible_ids:
#                 image_filename = f"{possible_id}.jpg"
#                 image_path = os.path.join(root_data_path, folder, image_filename)
#                 if os.path.exists(image_path):
#                     found_images.append(possible_id)
    
#         if len(found_images) == 1:
#             return found_images[0]  # Eindeutige ID und zugehöriger Ordner
#         elif len(found_images) > 1:
#             return found_images
#         return "not_found"  # Falls keine gefunden wird, ursprüngliche ID behalten
#     return image_id

# # IDs korrigieren
# df["image_id"] = df["image_id"].apply(find_corrected_id)

# # Aktualisierte CSV speichern
# df.to_csv(os.path.join(root_data_path, metadata, corr_file_name_id_corr), index=False)


# Update v4 to v5
df_need_update = pd.read_csv(os.path.join(root_data_path, metadata, v4_file_name), usecols=["image_id", "road_scenery"])
df_updates = pd.read_csv(os.path.join(root_data_path, metadata, corr_file_name_update))

# df_empty = pd.DataFrame(columns=["image_id", "road_scenery"])
df_remove = df_updates[df_updates["original_folder"]!="v1"]
df_updates = df_updates[df_updates["original_folder"]=="v1"]
df_updates = df_updates[df_updates["road_scenery_corr"].notna()]

df_need_update = df_need_update[~df_need_update["image_id"].isin(df_remove["image_id"].values)]

# Updates mit iterrows anwenden
for _, row in df_updates.iterrows():
    image_id, new_folder = row["image_id"], folder_map[row["road_scenery_corr"]]
    if image_id in df_need_update["image_id"].values:
        df_need_update.loc[df_need_update["image_id"] == image_id, "road_scenery"] = new_folder
    else:
        df_need_update = pd.concat([df_need_update, pd.DataFrame([[image_id, new_folder]], columns=["image_id", "road_scenery"])], ignore_index=True)


df_need_update.replace(to_replace="1_3_pedestrian__1_3_railway_platform", value="1_3_pedestrian__1_3_footway", inplace=True)
df_need_update.replace(to_replace="1_1_road__1_1_rails_on_road", value="1_1_road__1_1_road_general", inplace=True)
df_need_update = df_need_update[df_need_update["road_scenery"]!="2_1_no_focus_no_street__2_1_surface_covered"]

# Aktualisierte CSV speichern
df_need_update = df_need_update.sort_values(by="image_id").reset_index(drop=True)
df_need_update.to_csv(os.path.join(root_data_path, metadata, v5_file_name), index=False)