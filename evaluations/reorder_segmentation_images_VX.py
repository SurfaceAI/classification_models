import sys
sys.path.append('.')

import pandas as pd
from sklearn.metrics import precision_score, recall_score
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from experiments.config import global_config

ds_version = "V11"
root_data = global_config.global_config.get("root_data")
data_path = os.path.join(root_data, ds_version)

annot = pd.read_csv(os.path.join(data_path, "metadata", "annotations_combined.csv"))
annot = annot[["image_id", "roadtype"]]
annot["roadtype"].fillna("unspecified", inplace=True)
annot["image_id"] = annot["image_id"].astype(str)

segmentations_path = os.path.join(data_path, "segmentation", "seg_sel_func_max_area_in_lower_half_crop")

for root, _, fnames in sorted(os.walk(segmentations_path, followlinks=True)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        file_name = os.path.split(path)[-1]
        id, value = os.path.splitext(file_name)[0].split("_", 1)

        new_segmentation_path = os.path.join(segmentations_path, annot[annot["image_id"] == id]["roadtype"].values[0], value)
        if not os.path.exists(new_segmentation_path):
            os.makedirs(new_segmentation_path)

        shutil.move(path, os.path.join(new_segmentation_path, file_name))
        


