# Classification models

This repo contains the code to train models with the [StreetSurfaceVis dataset](https://zenodo.org/records/11449977) to predict the surface type and quality of street view images, and to evaluate the predictions.

## Data preparation

1. Download the [StreetSurfaceVis dataset](https://zenodo.org/records/11449977). Image size 1024 is recommended.
2. Store or link the images to this repo in the folder /data/V1_0/s_1024
3. Store the metadata file streetSurfaceVis_v1_0.csv in the folder /data/V1_0/metadata
4. Split the images to train and test data sets using [src/utils/prepare_train_test_folders_type_quality.py](src/utils/prepare_train_test_folders_type_quality.py)

## Training 

1. Adapt training settings in the configurations file [experiments/config/train_config.py](experiments/config/train_config.py).

2. Use [experiments/training_CC.py](experiments/training_CC.py) to train the surface type model and surface quality models per type. 

Note: Set `save_state = True` in the global config [experiments/config/global_config.py](experiments/config/global_config.py) to save the trained models. Set `wandb_on=True` and `wandb_mode=constants.WANDB_MODE_ON` if you want to track the training using "Weights & Biases".

## Prediction

1. Adapt prediction settings in the configurations file [experiments/config/predict_config.py](experiments/config/predict_config.py). Insert the model files for surface type and quality per type from your training.

2. Use [experiments/prediction_CC.py](experiments/prediction_CC.py) to predict the surface type and based on the predicted type the surface quality.

## Evaluation

1. Use the notebook [evaluations/analyze_prediction_V1_0.ipynb] to evaluate the predictions. Insert the prediction csv-file from your prediction.

## Folder structure

.
├── data/
│   └── V1_0/
│   │   ├── metadata/
│   │   │   └── streetSurfaceVis_v1_0.csv
│   │   ├── s_1024/
│   │   │   ├── 100609302385218.jpg
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── asphalt/
│   │   │   └── ...
│   │   └── train/
│   │   │   ├── asphalt/
│   │   │   └── ...
├── evaluations/
│   └── analyze_prediction_V1_0.ipynb
├── experiments/
│   ├── config/
│   │   ├── global_config.py
│   │   ├── predict_config.py
│   │   └── train_config.py
│   ├── prediction_CC.py
│   └── training_CC.py
├── prediction/
├── src/
│   ├── architecture/
│   │   ├── efficientnet.py
│   ├── models/
│   │   ├── prediction.py
│   │   └── training.py
│   ├── utils/
│   │   ├── checkpointing.py
│   │   ├── create_valid_dataset.py
│   │   ├── helper.py
│   │   ├── prepare_train_test_folders_type_quality.py
│   │   └── preprocessing.py
│   └── constants.py
├── trained_models
├── requirements.txt
└── README.md
