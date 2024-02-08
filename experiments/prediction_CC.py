import sys
sys.path.append('.')

import os
from src.utils import preprocessing
from src import constants
from src.models import prediction
from experiments.config import general_config


name = "test_RatekeCNN_VGG16_prediction"

# level or defined by input trained_models?
# level = constants.FLATTEN # constants.SMOOTHNESS (= CC?) # constants.SURFACE

# model name in trained model saveing
model_dict = {'trained_model': 'surface-rateke-20240207_202104-gnzhpn11_epoch0.pt',
                'submodels': {constants.ASPHALT: {'trained_model': 'smoothness-asphalt-vgg16-20240207_202414-krno5gva_epoch0.pt'},
                            constants.CONCRETE: {'trained_model': 'smoothness-concrete-vgg16-20240207_202524-jqetza3o_epoch0.pt'},
                                                },}


data_root = general_config.data_training_path

dataset = 'V0/predicted'

model_root = general_config.trained_model_path

predict_dir = os.path.join(general_config.data_training_path, 'prediction')

# TODO: predefine transformation for inference
transform = {
    "resize": constants.H256_W256,
    # "crop": constants.CROP_LOWER_MIDDLE_THIRD,
    "normalize": (constants.V4_ANNOTATED_MEAN, constants.V4_ANNOTATED_SD),
}
transform = preprocessing.transform(**transform)

# def run_config_and_prediction():
#     pass

gpu_kernel = 1

batch_size = 8

prediction.run_dataset_prediction(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size)
