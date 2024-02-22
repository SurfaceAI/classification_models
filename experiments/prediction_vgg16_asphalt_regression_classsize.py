import sys

sys.path.append(".")

from src.models import prediction
from experiments.config import global_config
from src import constants as const

# crop lower third + seed 42
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
    50: "asphalt-vgg16-20240221_184137-5bzhmros_epoch19.pt",
    100: "asphalt-vgg16-20240221_184631-lembdujr_epoch17.pt",
    200: "asphalt-vgg16-20240221_185237-47y70wch_epoch19.pt",
    400: "asphalt-vgg16-20240221_190048-7o952wtt_epoch14.pt",
    600: "asphalt-vgg16-20240221_191317-4bffchh1_epoch7.pt",
    'Inf': "asphalt-vgg16-20240221_192424-tzcfmtkv_epoch17.pt",
}

for max_class_size, model in model_names.itmens():
    config = {
        **base_config,
        "name": f"asphalt_c_third_42_{max_class_size}_prediction",
        "dataset": f"V7/annotated/asphalt",
        "model_dict": {"trained_model": model}, 
    }
    prediction.run_dataset_predict_csv(config)

# crop lower half + seed 42
base_config = {
    **global_config.global_config,
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V7_ANNOTATED_ASPHALT_MEAN, const.V7_ANNOTATED_ASPHALT_SD),
    },
    "batch_size": 96,
}

model_names = {
    50: "asphalt-vgg16-20240221_175440-fdwackzq_epoch19.pt",
    100: "asphalt-vgg16-20240221_175916-2beq8xjb_epoch8.pt",
    200: "asphalt-vgg16-20240221_180327-crjc7w4r_epoch5.pt",
    400: "asphalt-vgg16-20240221_180805-6clvhbnv_epoch5.pt",
    600: "asphalt-vgg16-20240221_181504-si1vn2kp_epoch2.pt",
    'Inf': "asphalt-vgg16-20240221_182152-m9cjnxez_epoch6.pt",
}

for max_class_size, model in model_names.itmens():
    config = {
        **base_config,
        "name": f"asphalt_c_half_42_{max_class_size}_prediction",
        "dataset": f"V7/annotated/asphalt",
        "model_dict": {"trained_model": model}, 
    }
    prediction.run_dataset_predict_csv(config)

# crop lower half + seed 1024
base_config = {
    **global_config.global_config,
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_HALF,
        "normalize": (const.V7_ANNOTATED_ASPHALT_MEAN, const.V7_ANNOTATED_ASPHALT_SD),
    },
    "batch_size": 96,
}

model_names = {
    50: "asphalt-vgg16-20240221_211609-jiba8rrc_epoch19.pt",
    100: "asphalt-vgg16-20240221_212036-luifjke6_epoch15.pt",
    200: "asphalt-vgg16-20240221_212607-e1et368q_epoch8.pt",
    400: "asphalt-vgg16-20240221_213145-2drlpy5x_epoch4.pt",
    600: "asphalt-vgg16-20240221_213814-tiudsxtf_epoch2.pt",
    'Inf': "asphalt-vgg16-20240221_214453-lw5zyx6p_epoch3.pt",
}

for max_class_size, model in model_names.itmens():
    config = {
        **base_config,
        "name": f"asphalt_c_half_1024_{max_class_size}_prediction",
        "dataset": f"V7/annotated/asphalt",
        "model_dict": {"trained_model": model}, 
    }
    prediction.run_dataset_predict_csv(config)
