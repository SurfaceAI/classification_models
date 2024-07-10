import sys
sys.path.append('.')

import pandas as pd

from src.models import prediction
from experiments.config import predict_config

df = pd.DataFrame(
    {"kernel": [0, 5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11],
     "sigma": [0, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5],
     "model": [
         "surface-efficientNetV2SLinear-20240627_104026-tushaj67_epoch8.pt",
         "surface-efficientNetV2SLinear-20240627_111538-sy6m4em5_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_115608-i08fr2sl_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_123620-iurfxap4_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_131448-rwirb1dg_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_135435-jaaaljs1_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_143406-ndho80fs_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_151256-pph3xz5y_epoch8.pt",
         "surface-efficientNetV2SLinear-20240627_155318-g9eqxhoy_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_163315-4r4xam8g_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_171302-4hz1btvo_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_175249-9jux9wlv_epoch9.pt",
         "surface-efficientNetV2SLinear-20240627_183300-at9kpvfr_epoch9.pt",
     ]},
)

config = predict_config.blur_V1_0

for index_m, row_m in df.iterrows():
    config["model_dict"]["trained_model"] = row_m["model"]
    for index_d, row_d in df.iterrows():
        config["transform"]["gaussian_blur_kernel"] = row_d["kernel"] if row_d["kernel"] != 0 else None
        config["transform"]["gaussian_blur_sigma"] = row_d["sigma"] if row_d["sigma"] != 0 else None
        config["name"] = f"effnet_blur_surface_pred_model{index_m}_dataset{index_d}"

        # prediction.run_dataset_predict_csv(predict_config.blur_V1_0)
        prediction.run_dataset_predict_csv(config)
