# In[1]:


import os

from functions import preprocessing, fix_seeds, compute_metrics, print_metrics, compute_splits_hash, run_hier_cnn

import sys
import shutil
import time
import signal
import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import GroupShuffleSplit

# Sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter


# In[3]:


# Sacred Configuration
ex = Experiment('BenelliHierOrd', interactive=False)
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Do not keep track of prints and stdout, stderr and so on.
SETTINGS.CAPTURE_MODE = 'no'

# In[4]:


@ex.config
def cfg():
    # Random seed
    seed = 3

    # Base path
    base_path = Path('/mnt/batch/tasks/shared/LS_root/mounts/clusters/gputrain/code/Users/S1095585/BenelliLegni')

    # Type of model that will be used
    model_name = "vgg16"

    # Shape of each image
    img_shape = (270, 470)

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
        'type': 'MAE',
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
        'enabled': False,
        'link': 'logit',
        'min_distance': 0.0,
        'use_slope': False,
        'fixed_thresholds': False
    }

    # OBD config
    obd = {
        'name': 'obd',
        'enabled': True
    }


    # Augmentation
    augment = True
    
    # Results path
    results_path = './results/results_hier_ord.csv'


# In[5]:


@ex.automain
@LogFileWriter(ex)
def run(seed, base_path, model_name, img_shape, trainable_convs, shared_layers,
                optimiser_params, loss_config, loss_config2, clm, obd, augment, results_path):
    
    # Set memory growth for gpus
    gpu_devices = tf.config.list_physical_devices('GPU')
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    # Fix random seeds
    fix_seeds(seed)

    # Load csv that contains data info
    # path to images folder
    path_imgs = base_path / 'CALCIO_CROP'
    # path to annotations file
    csv = pd.read_csv(base_path / '20201102_ExportDB.txt', sep=";") #20201102_ExportDB.txt

    # Load preprocessed data
    X, y = preprocessing(path_imgs, csv, 'CALCIO', 'rgb',
                         img_shape[0], img_shape[1], False, model_name)

    labels = np.unique(y[:, 0])
    n_labels = len(labels)

    # Handle Ctrl-C and SIGTERM signals
    def sigterm_handle(_signo, _stack_frame):
        print("Stopping...")
        shutil.rmtree(temp_dir)
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handle)
    signal.signal(signal.SIGINT, sigterm_handle)

    # Split train and test using group shuffle split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    gss_splits = list(gss_test.split(X=X, y=y, groups=y[:, 3]))
    train_idx, test_idx = gss_splits[0]

    # Compute splits md5sum to uniquely identify these splits
    test_gss_hash = compute_splits_hash(gss_splits)

    # Get train and test splits
    X_trainval, X_test = X[train_idx], X[test_idx]
    y_trainval, y_test = y[train_idx], y[test_idx]

    print(f"{y_trainval.shape=}, {y_test.shape=}")

    # Split train and validation using gss
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    gss_val_splits = list(gss_val.split(
        X=X_trainval, y=y_trainval, groups=y_trainval[:, 3]))
    train_idx, val_idx = gss_val_splits[0]

    # Get train and validation splits
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # Compute splits md5sum to uniquely identify these splits
    val_gss_hash = compute_splits_hash(gss_val_splits)

    
    print(f"{y_train.shape=}, {y_val.shape=}")


    start = time.time()

    # Initialise, run and predict
    test_pred_major, test_pred_minor = run_hier_cnn(train_data=(X_train, y_train), validation_data=(X_val, y_val),
                                              test_data=(X_test, y_test), clm=clm, obd=obd, optimiser_params=optimiser_params, 
                                                  loss_config=loss_config, loss_config2=loss_config2,
                                                   augment=augment, trainable_convs=trainable_convs, shared_layers=shared_layers,
                                                   seed=seed)

    end = time.time()
    total_time = end - start

    major_metrics = compute_metrics(y_test[:, 1], test_pred_major, num_classes=len(np.unique(y_test[:, 1])))
    print_metrics(major_metrics)

    test_pred_matrix = np.array(test_pred_major, dtype=int).reshape(-1, 1)

    minor_metrics = compute_metrics(y_test[:, 2], test_pred_minor, num_classes=len(np.unique(y_test[:, 2])))
    print_metrics(minor_metrics)

    test_pred_matrix = np.hstack((test_pred_matrix, test_pred_minor.reshape(-1, 1)))
    test_pred_matrix = test_pred_matrix.astype('uint8')

    print(test_pred_matrix.shape)

    # Inverse labels mapping to convert major,minor to global label
    # First level: major, second level: minor, value: global
    labels_inv_mapping = {0 : {0 : 0}, 1 : {0: 1, 1: 2, 2: 3}, 2: {0: 4, 1: 5, 2: 6}, 3: {0: 7, 1: 8, 2: 9}}

    test_pred_list = []
    for i in range(test_pred_matrix.shape[0]):
        major = test_pred_matrix[i,0]
        if major == 0:
            test_pred_list.append(0)
            continue
        minor = test_pred_matrix[i,1]
        test_pred_list.append(labels_inv_mapping[major][minor])

    final_test_pred = np.array(test_pred_list)


    end = time.time()
    total_time = end - start

    test_metrics = compute_metrics(y_test[:, 0], final_test_pred, num_classes=n_labels)
    test_metrics['time'] = total_time
    print_metrics(test_metrics)

    # Create a temporary dir where output files will be stored until added as sacred artifacts
    # Use PID and time to get an unique directory name for each execution
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
    

    # Return the mean validation metric because this way is easier to find the best execution in omniboard
    # Test metrics go in an artifact
    return float(test_metrics['MAE'])
