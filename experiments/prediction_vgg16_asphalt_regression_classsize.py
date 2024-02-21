import sys

sys.path.append(".")

from src.models import prediction
from experiments.config import global_config
from src import constants as const

base_config = {
    **global_config.global_config,
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V7_ANNOTATED_ASPHALT_MEAN, const.V7_ANNOTATED_ASPHALT_SD),
    },
    "batch_size": 96,
}

model_names = {
    50: "",
    100: "",
    200: "",
    400: "",
    600: "",
    'Inf': "",
}

for max_class_size, model in model_names.itmens():
    config = {
        **base_config,
        "name": f"asphalt_{max_class_size}_prediction",
        "dataset": f"V7/annotated/asphalt",
        "model_dict": {"trained_model": model}, 
    }
    prediction.run_dataset_predict_csv(config)
