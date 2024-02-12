import sys

sys.path.append(".")

import os

from experiments.config import general_config
from src import constants as const
from src.models import prediction
from src.utils import preprocessing

# level or defined by input trained_models?
# level = constants.FLATTEN # constants.SMOOTHNESS (= CC?) # constants.SURFACE

model_names = {
    const.ASPHALT: "smoothness-asphalt-vgg16Regression-20240212_165011-ekutirv5_epoch3.pt",
    const.CONCRETE: "smoothness-concrete-vgg16Regression-20240212_165215-t3eqkcbv_epoch2.pt",
    const.PAVING_STONES: "smoothness-paving_stones-vgg16Regression-20240212_165305-em9getah_epoch2.pt",
    const.SETT: "smoothness-sett-vgg16Regression-20240212_165357-hqd886gh_epoch3.pt",
    const.UNPAVED: "smoothness-unpaved-vgg16Regression-20240212_165445-j0veqa9u_epoch0.pt",
}

for surface in const.SURFACES:
    dataset = f"V6/annotated/{surface}"
    name = f"{surface}_prediction"

    data_root = general_config.data_training_path

    model_dict = {"trained_model": model_names[surface]}

    model_root = general_config.trained_model_path

    predict_dir = os.path.join(general_config.data_training_path, "prediction")

    # TODO: predefine transformation for inference
    transform = {
        "resize": const.H256_W256,
        # "crop": constants.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
    }
    transform = preprocessing.transform(**transform)

    gpu_kernel = general_config.gpu_kernel

    batch_size = 8

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
