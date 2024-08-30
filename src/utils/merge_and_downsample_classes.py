import sys

sys.path.append(".")

import os
import shutil

from sklearn.model_selection import train_test_split

from src.utils import preprocessing

def downsample_all_classes(source_path, target_path, selected_classes, total_sample_size, seed):

    dataset = preprocessing.FlattenFolders(source_path, selected_classes=selected_classes)

    samples_remaining, _ = train_test_split(dataset.samples, train_size=total_sample_size, random_state=seed, stratify=dataset.targets)
    
    for sample in samples_remaining:

        rel_path = os.path.relpath(sample[0], source_path)
        target_file = os.path.join(target_path, rel_path)
        
        os.makedirs(os.path.split(target_file)[0], exist_ok=True)

        shutil.copy2(sample[0], target_file)

def merge_classes(source_path, selected_classes_list, merged_class):

    target_path = os.path.join(source_path, merged_class)

    for cls in selected_classes_list:
        cls_path = os.path.join(source_path, cls)
        for root, _, fnames in sorted(os.walk(cls_path, followlinks=True)):
            for fname in sorted(fnames):
                source_file = os.path.join(root, fname)
                rel_path = os.path.relpath(source_file, cls_path)
                target_file = os.path.join(target_path, rel_path)
                os.makedirs(os.path.split(target_file)[0], exist_ok=True)
                shutil.copy2(source_file, target_file)

def merge_and_downsample_classes(source_path, selected_classes, merged_class, total_sample_size, seed):

    target_path = os.path.join(source_path, merged_class)

    dataset = preprocessing.FlattenFolders(source_path, selected_classes=selected_classes)

    samples_remaining, _ = train_test_split(dataset.samples, train_size=total_sample_size, random_state=seed, stratify=dataset.targets)

    for sample in samples_remaining:

        rel_path = os.path.relpath(sample[0], source_path)
        target_path = os.path.join(target_path, rel_path)
        
        os.makedirs(os.path.split(target_path)[0], exist_ok=True)

        shutil.copy2(sample[0], target_path)


from experiments.config  import global_config
from src import constants
root_path = global_config.global_config.get("root_data")
selected_classes = global_config.global_config.get("selected_classes")

downsample_all_classes(os.path.join(root_path, "V1_0", "annotated"), os.path.join(root_path, "V1_0", "downsampled_rtk"), selected_classes, 6297, 42)

merge_classes(os.path.join(root_path, "V1_0", "downsampled_rtk"), ["asphalt", "concrete"], "asphalt-concrete")
merge_classes(os.path.join(root_path, "V1_0", "downsampled_rtk"), ["paving_stones", "sett"], "paving_stones-sett")

# merge_and_downsample_classes(os.path.join(root_path, "V1_0", "downsampled_rtk"), {key: selected_classes[key] for key in ["asphalt", "concrete"] if key in selected_classes}, "asphalt-concrete_quality", 3281, 42) 
