import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import signal
import sys
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from functions import fix_seeds, compute_metrics, print_metrics, compute_splits_hash, run_hier_cnn, run_cnn
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter
import wandb
from wandb.keras import WandbCallback

# Sacred Configuration
ex = Experiment('BenelliHierOrd', interactive=False)
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Do not keep track of prints and stdout, stderr and so on.
SETTINGS.CAPTURE_MODE = 'no'


def load_images_and_labels(base_path, img_shape, custom_label_order):
    images = []
    labels = []
    for label_folder in os.listdir(base_path):
        label_folder_path = os.path.join(base_path, label_folder)
        if os.path.isdir(label_folder_path):
            for img_file in os.listdir(label_folder_path):
                img_path = os.path.join(label_folder_path, img_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_shape)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label_folder)  # Use the folder name as the label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    # Create a mapping based on custom_label_order
    label_to_index = {label: idx for idx, label in enumerate(custom_label_order)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    # Map labels to the custom order indices
    y = np.array([label_to_index[label] for label in labels])
    
    return np.array(images), y, label_to_index, index_to_label

def fix_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# def compute_splits_hash(splits):
#     import hashlib
#     hash_md5 = hashlib.md5()
#     hash_md5.update(np.array(splits).tobytes())
#     return hash_md5.hexdigest()

def run(seed, base_path, model_name, use_wandb, img_shape, trainable_convs, shared_layers,
        optimiser_params, loss_config, loss_config2, clm, obd, augment, results_path):
    
    if use_wandb:
        wandb.init(project="road-surface-classification-ordinal-regression")

    # Set memory growth for gpus
    gpu_devices = tf.config.list_physical_devices('GPU')
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    # Fix random seeds
    fix_seeds(seed)


    custom_label_order = ['excellent', 'good', 'intermediate', 'bad']

    # Load images and labels
    X, y, label_to_index, index_to_label = load_images_and_labels(base_path, img_shape, custom_label_order)

    print("Unique labels before split:", np.unique(y))

    labels = np.unique(y)
    n_labels = len(labels)

    # Handle Ctrl-C and SIGTERM signals
    # def sigterm_handle(_signo, _stack_frame):
    #     print("Stopping...")
    #     shutil.rmtree(temp_dir)
    #     sys.exit(0)
    # signal.signal(signal.SIGTERM, sigterm_handle)
    # signal.signal(signal.SIGINT, sigterm_handle)

    # Split train and test using group shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    sss_splits = list(sss.split(X=X, y=y))
    train_idx, test_idx = sss_splits[0]

    # Compute splits md5sum to uniquely identify these splits
    # test_gss_hash = compute_splits_hash(gss_splits)

    # Get train and test splits
    X_trainval, X_test = X[train_idx], X[test_idx]
    y_trainval, y_test = y[train_idx], y[test_idx]
    
    print("Unique labels in y_trainval:", np.unique(y_trainval))
    print("Unique labels in y_test:", np.unique(y_test))


    # Further split the trainval set into training and validation sets
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)  # 0.25 * 0.8 = 0.2
    sss_splits_val = list(sss_val.split(X=X_trainval, y=y_trainval))
    train_idx, val_idx = sss_splits_val[0]

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    train_data = (X_train, y_train)
    validation_data = (X_val, y_val)
    test_data = (X_test, y_test)

    test_pred = run_cnn(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        optimiser_params=optimiser_params,
        clm=clm,
        obd=obd,
        loss_config=loss_config,
        trainable_convs=trainable_convs,
        labels=labels,
        return_labels=True,
        augment=True
    )
    
    test_metrics = compute_metrics(y_test[:, 1], test_pred, num_classes=len(np.unique(y_test[:, 1])))
    print_metrics(test_metrics)

    test_pred_matrix = np.array(test_pred, dtype=int).reshape(-1, 1)
    print(test_pred_matrix.shape)
    
    temp_dir = Path(f'./temp{os.getpid()}_{time.time()}')
    os.makedirs(temp_dir)

    # Save results to file and add them as artifacts
    with open(temp_dir / f'metrics.csv', 'w') as f:
        for key in test_metrics.keys():
            if key != 'Confusion matrix':
                f.write(f"test_{key},{round(test_metrics[key], 5)}\n")

    np.savetxt(temp_dir / 'test_confmat.txt',
               test_metrics['Confusion matrix'], fmt='%d')

    # Add as artifact
    ex.add_artifact(temp_dir / f'metrics.csv')
    ex.add_artifact(temp_dir / f'test_confmat.txt')

    # Remove temp folder
    shutil.rmtree(temp_dir)

    # Convert str to Path
    results_path_tot = Path(results_path)

    with open(results_path_tot, 'a') as f:
        # If file is empty, write the header
        if results_path_tot.stat().st_size == 0:
            f.write("seed,")
            for key in test_metrics.keys():
                if key != 'Confusion matrix':
                    f.write(f"Test-{key},")
            f.write("\n")

        # Write folds info
        f.write(f"{seed},")

        # Write test metrics
        for key in test_metrics.keys():
            if key != 'Confusion matrix':
                f.write(f"{round(test_metrics[key], 5)},")
        f.write('\n')
        
    if use_wandb:
        wandb.finish()



# Configuration function
def cfg():
    # Random seed
    seed = 3

    # Base path
    #base_path = Path(r'c:\Users\esthe\Documents\GitHub\classification_models\data\training\V12/annotated/asphalt')
    base_path = Path(r"/home/esther/surfaceai/classification_models/data/training/V12/annotated/asphalt")

    # Type of model that will be used
    model_name = "vgg16"
    
    #wand on or off
    use_wandb = True

    # Shape of each image
    img_shape = (224, 224)

    # Are the convolutional layers trainable?
    trainable_convs = False

    # Level of shared layers
    shared_layers = 'All' #All or 2ConvBlocks

    # Optimiser params
    optimiser_params = {
        'lr': 0.01,
        'bs': 64,
        'epochs': 50
    }

    # Basically, QWK for CLM and MAE for OBD

    # Loss config for macro task
    loss_config = {
        'type': 'CCE',
        'weight': 0.5
    }

    # Loss config for micro task
    loss_config2 = {
        'type': 'MAE',
        'weight': 0.5
    }

    # If CLM is enabled, OBD must be disabled and vice versa

    # CLM config
    clm = {
        'name': 'clm',
        'enabled': True,
        'link': 'logit',
        'min_distance': 0.0,
        'use_slope': False,
        'fixed_thresholds': False
    }

    # OBD config
    obd = {
        'name': 'obd',
        'enabled': False
    }

    # Augmentation
    augment = True
    
    # Results path
    results_path = './results/results_hier_ord.csv'

    return seed, base_path, model_name, use_wandb, img_shape, trainable_convs, shared_layers, optimiser_params, loss_config, loss_config2, clm, obd, augment, results_path

