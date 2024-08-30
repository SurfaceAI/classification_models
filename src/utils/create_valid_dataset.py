import sys

sys.path.append(".")

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.config import train_config
from src.utils import preprocessing

def save_train_valid_split(config):

    data_path = os.path.join(config.get("root_data"), config.get("dataset"))
    metadata_path = os.path.join(config.get("root_data"), config.get("metadata"))
    selected_classes = config.get("selected_classes")
    seed = config.get("seed")
    print(seed)
    validation_size = config.get("validation_size")

    dataset = preprocessing.FlattenFolders(data_path, selected_classes=selected_classes)

    id_list = []
    label_list = []
    for image_path, label_idx in dataset.samples:
        id = os.path.splitext(os.path.split(image_path)[-1])[0]
        label = dataset.classes[label_idx]
        id_list.append(id)
        label_list.append(label)

    trainset, validset = train_test_split(id_list, test_size=validation_size, random_state=seed, stratify=label_list)

    df = pd.DataFrame()
    df["image_id"] = id_list
    df["label"] = label_list
    df["is_in_validation"] = [id in validset for id in id_list]

    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    df.to_csv(os.path.join(metadata_path, config.get("train_valid_split_list")), index=False)


if __name__ == "__main__":

    config=train_config.train_valid_split_params_rtk

    save_train_valid_split(config=config)