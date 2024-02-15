import sys

sys.path.append(".")

import os

from experiments.config import global_config
from src import constants as const
from src.models import prediction
from src.utils import preprocessing

# level or defined by input trained_models?
# level = constants.FLATTEN # constants.SMOOTHNESS (= CC?) # constants.SURFACE
data_root = global_config.global_config.get("root_data")
model_root = global_config.global_config.get("root_model")
predict_dir = os.path.join(
    global_config.global_config.get("root_data"), "prediction"
)

gpu_kernel = global_config.global_config.get("gpu_kernel")

batch_size = 48

# TODO: predefine transformation for inference
transform = {
    "resize": const.H256_W256,
    "crop": const.CROP_LOWER_MIDDLE_THIRD,
    "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
}
transform = preprocessing.transform(**transform)

dataset = "V6/annotated"
name = "surface_prediction"

model_dict = {"trained_model": ""}

prediction.run_dataset_prediction_csv(
    name,
    data_root,
    dataset,
    transform,
    model_root,
    model_dict,
    predict_dir,
    gpu_kernel,
    batch_size,
)
