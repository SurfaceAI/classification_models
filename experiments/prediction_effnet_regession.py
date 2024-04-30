import sys

sys.path.append(".")

from src.models import prediction
from experiments.config import global_config
from src import constants as const

base_config = {
    **global_config.global_config,
    "transform": {
        "resize": (384, 384),
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V9_ANNOTATED_MEAN, const.V9_ANNOTATED_SD),
    },
    "batch_size": 16,
}

model_names = {
    const.ASPHALT: "asphalt-efficientNetV2SLinear-20240318_140538-58wabd24_epoch7.pt",
    const.CONCRETE: "concrete-efficientNetV2SLinear-20240318_141908-d6xa76up_epoch7.pt",
    const.PAVING_STONES: "paving_stones-efficientNetV2SLinear-20240318_142324-36lsqmb9_epoch7.pt",
    const.SETT: "sett-efficientNetV2SLinear-20240318_142953-np4ap09r_epoch5.pt",
    const.UNPAVED: "unpaved-efficientNetV2SLinear-20240318_143640-ijawnat4_epoch5.pt",
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
        "name": f"all_train_{surface}_prediction",
        "dataset": f"V11/annotated/{surface}",
        "model_dict": {"trained_model": model_names[surface]}, 
    }
    prediction.run_dataset_predict_csv(config)
