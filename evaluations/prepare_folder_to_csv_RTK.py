import sys

sys.path.append(".")

from src import constants as const
from experiments.config  import global_config

import pandas as pd
import os
from torchvision import datasets
from PIL import Image

data_path = global_config.global_config.get('root_data')
dataset = "RTK/GT"
metadata = "RTK/metadata"
file_name = 'GT_RTK.csv'

# selected_classes = {
#         'asphalt': [
#             'good',
#             'regular',
#             'bad',
#         ],
#         'paved': [
#             'good',
#             'regular',
#             'bad',
#         ],
#         'unpaved': [
#             'regular',
#             'bad',
#         ],
#     }

# valid_labels = []
# for key, values in selected_classes.items():
#     for value in values:
#         label = key + '__' + value
#         valid_labels.append(label)

columns = ['image_id', 'type_true', 'quality_true']
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
                label_type, label_quality = folders.split("/")
                i = df.shape[0]
                df.loc[i, columns] = [image_id, label_type, label_quality]
os.makedirs(os.path.join(data_path, metadata), exist_ok=True)
df.to_csv(os.path.join(data_path, metadata, file_name), index=False)

