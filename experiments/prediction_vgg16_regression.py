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
        "normalize": (const.V4_ANNOTATED_MEAN, const.V4_ANNOTATED_SD),
    },
    "batch_size": 96,
}

model_names = {
    const.ASPHALT: "smoothness-asphalt-vgg16Regression-20240212_165011-ekutirv5_epoch3.pt",
    const.CONCRETE: "smoothness-concrete-vgg16Regression-20240212_165215-t3eqkcbv_epoch2.pt",
    const.PAVING_STONES: "smoothness-paving_stones-vgg16Regression-20240212_165305-em9getah_epoch2.pt",
    const.SETT: "smoothness-sett-vgg16Regression-20240212_165357-hqd886gh_epoch3.pt",
    const.UNPAVED: "smoothness-unpaved-vgg16Regression-20240212_165445-j0veqa9u_epoch0.pt",
}

for surface in [
    const.ASPHALT,
    const.CONCRETE,
    const.PAVING_STONES,
    const.SETT,
    const.UNPAVED,
]:
    config = {
        **base_config,
        "name": f"{surface}_prediction",
        "dataset": f"V6/annotated/{surface}",
        "model_dict": {"trained_model": model_names[surface]}, 
    }
    prediction.run_dataset_predict_csv(config)
