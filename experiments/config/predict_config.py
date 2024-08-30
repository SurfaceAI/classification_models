from src import constants as const
from experiments.config import global_config

model_dict_old = {
    # "trained_model": "surface-efficientNetV2SLinear-20240805_142047-5bvhwua5_epoch6.pt",
    #for test only!!!
    "trained_model": "surface-efficientNetV2SLinear-20240828_163454-kafm6f5f_epoch0.pt",
    # crop rtk
    # "trained_model": "surface-efficientNetV2SLinear-20240812_144738-tzdnjsl9_epoch7.pt",
    "level": const.TYPE,
    "submodels": {
        const.ASPHALT: {
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240805_150657-vbh3etku_epoch19.pt",
            # crop rtk
            # "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240812_154148-5wgu932t_epoch15.pt",
            "level": const.QUALITY,
        },
        const.CONCRETE: {
            "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240805_153157-6lg00blf_epoch16.pt",
            # crop rtk
            # "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240812_160659-vkqldetk_epoch16.pt",
            "level": const.QUALITY,
        },
        const.PAVING_STONES: {
            "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240805_153933-sag52zdy_epoch17.pt",
            # crop rtk
            # "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240812_161419-jdxs5uu8_epoch15.pt",
            "level": const.QUALITY,
        },
        const.SETT: {
            "trained_model": "smoothness-sett-efficientNetV2SLinear-20240805_155449-jj2sy16u_epoch18.pt",
            # crop rtk
            # "trained_model": "smoothness-sett-efficientNetV2SLinear-20240812_162958-mc41wvya_epoch13.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240805_160525-bfdgx097_epoch19.pt",
            # crop rtk
            # "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240812_164111-0v1ugvoi_epoch16.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict = {
    "trained_model": "surface-efficientNetV2SLinear-20240814_235454-fz95dogh_epoch11.pt",
    "level": const.TYPE,
    "submodels": {
        "asphalt-concrete": {
            "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240815_010429-us92f3mk_epoch16.pt",
            "level": const.QUALITY,
        },
        "paving_stones-sett": {
            "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240815_014418-nlqovqi5_epoch10.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240815_020818-9t5wtv39_epoch17.pt",
            "level": const.QUALITY,
        },
    },
}

# model_dict_rtk = {
#     # crop rtk
#     "trained_model": "surface-efficientNetV2SLinear-20240809_095944-39el2iwq_epoch7.pt",
#     # crop lower middle half
#     # "trained_model": "surface-efficientNetV2SLinear-20240809_104410-5c2khyzs_epoch7.pt",
#     "level": const.TYPE,
#     "submodels": {
#         const.ASPHALT: {
#             # crop rtk
#             "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240809_101420-3eqgebdx_epoch18.pt",
#             # crop lower middle half
#             # "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240809_105919-duc7cpjh_epoch16.pt",
#             "level": const.QUALITY,
#         },
#         const.PAVED: {
#             # crop rtk
#             "trained_model": "smoothness-paved-efficientNetV2SLinear-20240809_102727-jvs8ddbd_epoch16.pt",
#             # crop lower middle half
#             # "trained_model": "smoothness-paved-efficientNetV2SLinear-20240809_111940-2tly9cul_epoch16.pt",
#             "level": const.QUALITY,
#         },
#         const.UNPAVED: {
#             # crop rtk
#             "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240809_102934-mvzaondi_epoch15.pt",
#             # crop lower middle half
#             # "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240809_112402-hux89no4_epoch18.pt",
#             "level": const.QUALITY,
#         },
#     },
# }

model_dict_rtk = {
    # crop lower middle half
    "trained_model": "surface-efficientNetV2SLinear-20240814_175945-46npupmv_epoch5.pt",
    "level": const.TYPE,
    "submodels": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240814_182112-ur7qij9l_epoch19.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240814_184123-f22wvik6_epoch18.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240814_185347-y5erwqth_epoch15.pt",
            "level": const.QUALITY,
        },
    },
}

default_params = {
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
}

CC_V1_0_train = {
    **global_config.global_config,
    **default_params,
    # "name": "efficientnet_CC_prediction",
    # "model_dict": model_dict,
    "name": "RTK_complete_lmh-crop_trained_prediction",
    "model_dict": model_dict_rtk,
    "dataset": "V1_0/annotated",
    # "transform": {
    #     **default_params.get("transform"),
    #     "crop": "lower_half_rtk",
    # },
}

CC_V1_0_test = {
    **global_config.global_config,
    **default_params,
    "name": "efficientnet_CC_prediction",
    "model_dict": model_dict,
    "dataset": "V1_0/test",
}

CC_RTK = {
    **global_config.global_config,
    **default_params,
    # "name": "efficientnet_CC_prediction",
    # "model_dict": model_dict,
    "name": "V1_0_lmh-crop_trained_prediction",
    "model_dict": model_dict_old,
    "dataset": "RTK/GT",
    # "transform": {
    #     **default_params.get("transform"),
    #     "crop": "lower_half_rtk",
    #     # "crop": None,
    # },
}
