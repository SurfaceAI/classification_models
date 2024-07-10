import sys
sys.path.append('.')

import pandas as pd

from src.models import prediction
from experiments.config import predict_config

# # max 10 epochs
# df = pd.DataFrame(
#     {"kernel": [0, 5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11],
#      "sigma": [0, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5],
#      "model": [
#          "surface-efficientNetV2SLinear-20240627_104026-tushaj67_epoch8.pt",
#          "surface-efficientNetV2SLinear-20240627_111538-sy6m4em5_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_115608-i08fr2sl_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_123620-iurfxap4_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_131448-rwirb1dg_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_135435-jaaaljs1_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_143406-ndho80fs_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_151256-pph3xz5y_epoch8.pt",
#          "surface-efficientNetV2SLinear-20240627_155318-g9eqxhoy_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_163315-4r4xam8g_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_171302-4hz1btvo_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_175249-9jux9wlv_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240627_183300-at9kpvfr_epoch9.pt",
#      ]},
# )

# config = predict_config.blur_weseraue

# for index_m, row_m in df.iterrows():
#     config["model_dict"]["trained_model"] = row_m["model"]

#     config["name"] = f"effnet_blur_surface_pred_weseraue_model{index_m}"

#     prediction.run_dataset_predict_csv(config)

# full training
df = pd.DataFrame(
    {"kernel": [0, 5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11],
     "sigma": [0, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5, 2, 3.5, 5],
     "model": [
         "surface-efficientNetV2SLinear-20240610_185408-j3ob3p5o_epoch6.pt",
         "surface-efficientNetV2SLinear-20240702_223604-wwxwh5yt_epoch9.pt",
         "surface-efficientNetV2SLinear-20240702_234743-13gdut82_epoch8.pt",
         "surface-efficientNetV2SLinear-20240703_005450-w6si80d5_epoch9.pt",

         "surface-efficientNetV2SLinear-20240703_015631-9t6yogfu_epoch6.pt",
         "surface-efficientNetV2SLinear-20240703_024659-wmmu3h12_epoch7.pt",
         "surface-efficientNetV2SLinear-20240703_034210-xe6z21s6_epoch12.pt",

         "surface-efficientNetV2SLinear-20240703_045616-p79a2us6_epoch9.pt",
         "surface-efficientNetV2SLinear-20240703_060036-w9fq8zka_epoch13.pt",
         "surface-efficientNetV2SLinear-20240703_071942-1qw7gsxk_epoch13.pt",

         "surface-efficientNetV2SLinear-20240703_083734-ogswwoxp_epoch6.pt",
         "surface-efficientNetV2SLinear-20240703_092905-mq717zg1_epoch5.pt",
         "surface-efficientNetV2SLinear-20240703_101751-rc6mu56j_epoch9.pt",
     ]},
)

config = predict_config.blur_weseraue

for index_m, row_m in df.iterrows():
    config["model_dict"]["trained_model"] = row_m["model"]

    config["name"] = f"effnet_blur_surface_pred_weseraue_full_model{index_m}"

    prediction.run_dataset_predict_csv(config)
