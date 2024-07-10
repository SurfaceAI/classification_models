import sys
sys.path.append('.')

import pandas as pd

from src.models import prediction
from experiments.config import predict_config

# # epochs of high blur
# df = pd.DataFrame(
#     {"epoch": [6, 7, 10, 9, 13, 15],
#      "lr": [3, 3, 3, 1, 1, 1],
#      "model": [
#          "surface-efficientNetV2SLinear-20240702_220853-yr8d9jc0_epoch5.pt",
#          "surface-efficientNetV2SLinear-20240702_220853-yr8d9jc0_epoch6.pt",
#          "surface-efficientNetV2SLinear-20240702_220853-yr8d9jc0_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240702_231810-j1jf552j_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240702_231810-j1jf552j_epoch13.pt",
#          "surface-efficientNetV2SLinear-20240702_231810-j1jf552j_epoch15.pt",
#      ]},
# )

# config = predict_config.blur_weseraue

# for _, row_m in df.iterrows():
#     config["model_dict"]["trained_model"] = row_m["model"]

#     config["name"] = f"effnet_high_blur_lr{row_m["lr"]}_surface_pred_weseraue_epoch{row_m["epoch"]}"

#     prediction.run_dataset_predict_csv(config)

# # high blur with different lr + multiple runs
# df = pd.DataFrame(
#     {"run": [1, 2, 3, 4, 5, 6, 7, 8],
#      "model": [
#          "surface-efficientNetV2SLinear-20240617_105759-nh5vboqz_epoch16.pt",
#          "surface-efficientNetV2SLinear-20240703_154029-w4gmpay5_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240702_231810-j1jf552j_epoch15.pt",
#          "surface-efficientNetV2SLinear-20240703_120024-8lnxq8fg_epoch15.pt",
#          "surface-efficientNetV2SLinear-20240627_183300-at9kpvfr_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240703_101751-rc6mu56j_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240702_220853-yr8d9jc0_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240703_105302-opvl4zy2_epoch9.pt",
#      ]},
# )

# config = predict_config.blur_V1_0

# for _, row_m in df.iterrows():
#     config["model_dict"]["trained_model"] = row_m["model"]

#     config["name"] = f"effnet_low_preresize_surface_pred_V1_0_run{row_m["run"]}"

#     prediction.run_dataset_predict_csv(config)

# # low blur with different lr + multiple runs
# df = pd.DataFrame(
#     {"run": [1, 2, 3, 4, 5, 6],
#      "model": [
#          "surface-efficientNetV2SLinear-20240703_184112-h61jasm3_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240703_200018-iek8gehp_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240703_210411-mtkct23v_epoch15.pt",
#          "surface-efficientNetV2SLinear-20240703_222846-cuqhiu88_epoch15.pt",
#          "surface-efficientNetV2SLinear-20240703_235518-06o1cf5f_epoch9.pt",
#          "surface-efficientNetV2SLinear-20240704_005713-cbmtjqtr_epoch6.pt",
#      ]},
# )

# config = predict_config.blur_weseraue

# for _, row_m in df.iterrows():
#     config["model_dict"]["trained_model"] = row_m["model"]

#     config["name"] = f"effnet_low_blur_surface_pred_weseraue__super_small_pano_run{row_m["run"]}"

#     prediction.run_dataset_predict_csv(config)

# NO blur with different lr + multiple runs
df = pd.DataFrame(
    {"run": [1, 2, 3, 4, 5, 6, 7],
     "model": [
         "surface-efficientNetV2SLinear-20240704_171224-58rcrh0o_epoch8.pt",
         "surface-efficientNetV2SLinear-20240704_180352-pbxwsbgv_epoch10.pt",
         "surface-efficientNetV2SLinear-20240704_190641-cbs2qha6_epoch10.pt",
         "surface-efficientNetV2SLinear-20240704_202334-h842rbjj_epoch10.pt",
         "surface-efficientNetV2SLinear-20240704_211831-ntvzab3t_epoch7.pt",
         "surface-efficientNetV2SLinear-20240704_220436-7v8p5y2o_epoch8.pt",
        #  "surface-efficientNetV2SLinear-20240610_185408-j3ob3p5o_epoch6.pt", # first model
     ]},
)

config = predict_config.blur_weseraue

for _, row_m in df.iterrows():
    config["model_dict"]["trained_model"] = row_m["model"]

    config["name"] = f"effnet_no_blur_surface_pred_weseraue__small_pano_run{row_m["run"]}"

    prediction.run_dataset_predict_csv(config)
