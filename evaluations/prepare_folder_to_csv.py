import sys

sys.path.append(".")

from src import constants as const
from experiments.config  import global_config

import pandas as pd
import os
from torchvision import datasets
from PIL import Image

data_path = global_config.global_config.get('data_path')
dataset = "V1_0/road_scenery"
metadata = "V1_0/metadata"
file_name = 'annotations_scenery.csv'

selected_classes = {
        '1_1_road': [
            '1_1_parking_area',
            '1_1_rails_on_road',
            '1_1_road_general',
        ],
        '1_2_cycleway': [
            '1_2_hochbord',
            '1_2_lane',
        ],
        '1_3_pedestrian': [
            '1_3_pedestrian_area',
            '1_3_railway_platform',
            '1_3_sidewalk',
        ],
        '1_4_path': [
            '1_4_path_unspecified',
            '1_4_trampling_trail',
        ],
        '2_1_no_focus': [
            '2_1_other',
            '2_1_vertical',
        ],
        '2_2_no_street': [
            '2_2_all'
        ],
    }

valid_labels = []
for key, values in selected_classes.items():
    for value in values:
        label = key + '__' + value
        valid_labels.append(label)

columns = ['image_id', 'road_scenery']
df = pd.DataFrame(columns=columns)

directory = os.path.join(data_path, dataset)

extensions = datasets.folder.IMG_EXTENSIONS
def is_valid_file(x):
    return datasets.folder.has_file_allowed_extension(x, extensions)

for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        if is_valid_file(path):
            try:
                Image.open(path)
            except:
                print(f'Corrupted image: {path}')
                continue
            else:
                image_id = os.path.splitext(fname)[0]
                folders = os.path.relpath(root, directory)
                label = folders.replace("/", "__")
                if label not in valid_labels:
                    continue
                i = df.shape[0]
                df.loc[i, columns] = [image_id, label]

df.to_csv(os.path.join(data_path, metadata, file_name))

