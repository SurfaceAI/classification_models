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
        "normalize": (const.V11_ANNOTATED_MEAN, const.V11_ANNOTATED_SD),
    },
    "batch_size": 16,
    "gpu_kernel": 0,
}

model_names = {
    const.ASPHALT: "asphalt-efficientNetV2SLinear-20240417_090042-y1unrg9v_epoch6.pt",
    const.CONCRETE: "concrete-efficientNetV2SLinear-20240417_091332-5gic9utg_epoch6.pt",
    const.PAVING_STONES: "paving_stones-efficientNetV2SLinear-20240417_091725-yku3dp8z_epoch4.pt",
    const.SETT: "sett-efficientNetV2SLinear-20240417_092325-9rzjnyqa_epoch6.pt",
    const.UNPAVED: "unpaved-efficientNetV2SLinear-20240417_092934-dmcitr2h_epoch6.pt",
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
