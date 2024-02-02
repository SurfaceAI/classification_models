├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── training       <- training data.
│   └── test           <- test data.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── trained_models (anstatt models?)            <- Trained and serialized models, model predictions, or model summaries
│   ├── efficientnet_surface   <-
│   │   ├── checkpoints
│   │   └── efficientnet_v1.pt
│   │
│   ├── vgg16_surface       <- Data from third party sources.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
│
│
├── evaluations        <- model evaluation
│                         
├── experiments        <- run_efficientnet.py / run_experiments.py
│                         
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── results            <- track model performance, wandb folder?
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py (?)   <- Makes src a Python module
│   │
│   ├── constants.py   <- 
│   │
│   ├── config         <- 
│   │   ├── train_config.py
│   │   └── general_config
│   │
│   ├── data (not here -> dataset_creation)          <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features (we don't have this)      <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_surface.py incl. main()
│   │   └── train_surface.py incl main()
│   │
│   ├── architecture           <- 
│   │   ├── vgg16.py   <- model classes? or in config? model_config in models?
│   │   └── efficientnet.py
│   │
│   ├── utils           <- 
│   │   ├── preprocessing.py
│   │   ├── param_config.py
│   │   ├── str_conv.py  <- model, optimizer, etc. name conversion to classes or dicts
│   │   └── helper.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
│
├── tests              <- Unit tests, integration tests, and validation tests contribute to the reliability and robustness of the project. Organized based │
│                         on their purpose or the specific components they target.
│
└── tox.ini (?)            <- tox file with settings for running tox; see tox.readthedocs.io