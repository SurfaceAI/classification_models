import sys

sys.path.append(".")

from src.models import prediction
from experiments.config import global_config
from src import constants as const

base_config = {
    **global_config.global_config,
    "transform": {
        **global_config.global_config.get("transform"),
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
}

dataset_V1_0 = "V1_0/annotated"

dataset_V1_0_downsampled_rtk = "V1_0/downsampled_rtk"

dataset_RTK = "RTK/GT"

dataset_V1_0_test = "V1_0/test"

model_dict_V1_0 = {
    "model_name": "V1_0_merged_lmh-crop",
    "42": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240815_180431-an81pm4m_epoch4.pt",
            "level": const.TYPE,
        },
        "quality": {
            "asphalt-concrete": {
                "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240815_184710-mpyc9ti6_epoch19.pt",
                "level": const.QUALITY,
            },
            "paving_stones-sett": {
                "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240815_192825-y4h49ajf_epoch10.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240815_195138-prwqzwdh_epoch17.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1024": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_125942-0sd8r7xe_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            "asphalt-concrete": {
                "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240820_134701-u7qfm75m_epoch18.pt",
                "level": const.QUALITY,
            },
            "paving_stones-sett": {
                "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240820_143000-sw18ipea_epoch16.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_150434-0z5uk12r_epoch19.pt",
                "level": const.QUALITY,
            },
        },
    },
    "3": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_201950-x6anlifp_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            "asphalt-concrete": {
                "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240820_211159-ets71c2u_epoch19.pt",
                "level": const.QUALITY,
            },
            "paving_stones-sett": {
                "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240820_215514-bo2cspl1_epoch14.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_222558-uzaf91pz_epoch15.pt",
                "level": const.QUALITY,
            },
        },
    },
    "57": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_223318-frswu00z_epoch9.pt",
            "level": const.TYPE,
        },
        "quality": {
            "asphalt-concrete": {
                "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240820_233847-vfdt0xc7_epoch18.pt",
                "level": const.QUALITY,
            },
            "paving_stones-sett": {
                "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240821_002209-rh1repx3_epoch17.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_005302-lupqtzi6_epoch14.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1000": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240821_010024-jsm43nw8_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            "asphalt-concrete": {
                "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240821_014919-7dqcovte_epoch17.pt",
                "level": const.QUALITY,
            },
            "paving_stones-sett": {
                "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240821_023253-w5g7sn8l_epoch13.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_030332-0dlgjmx5_epoch5.pt",
                "level": const.QUALITY,
            },
        },
    },
}

model_dict_V1_0_annotated = {
    "model_name": "V1_0_annotated",
    "42": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_183302-0o1irfvo_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_200440-yistokt5_epoch17.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240820_205727-tqo5amc9_epoch15.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240820_210918-1l1qq3r8_epoch9.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240820_213605-0r6ev91o_epoch16.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_215344-9vay3bas_epoch17.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1024": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_220340-yv43qf4n_epoch4.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_231838-c7gcbmh1_epoch18.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240821_000458-88pwyowt_epoch7.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240821_001312-lo6ckuok_epoch15.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240821_004643-8w9cngpc_epoch16.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_010415-ntegrlp1_epoch19.pt",
                "level": const.QUALITY,
            },
        },
    },
    "3": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240821_011412-jxcz6hij_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240821_023540-d04vefqv_epoch18.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240821_032040-j0dr4tyu_epoch17.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240821_033112-hprrnhj6_epoch16.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240821_040221-phhr0x5e_epoch10.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_041547-myuttgnt_epoch18.pt",
                "level": const.QUALITY,
            },
        },
    },
    "57": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240821_042421-z6agxmlq_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240821_054222-58x7i3lu_epoch19.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240821_062531-qnfesgfs_epoch19.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240821_063601-co1a7s3o_epoch18.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240821_070605-mca0uj5h_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_072134-2qyydhlt_epoch13.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1000": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240821_072958-xxcl8lop_epoch3.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240821_083426-t3f4r26l_epoch16.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-efficientNetV2SLinear-20240821_091804-saho27ue_epoch13.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-efficientNetV2SLinear-20240821_093323-vkucjr87_epoch14.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-efficientNetV2SLinear-20240821_101503-gdwsznfn_epoch15.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240821_103337-933lvua0_epoch16.pt",
                "level": const.QUALITY,
            },
        },
    },
}

model_dict_V1_0_blur = {
    "model_name": "V1_0_merged_lmh-crop_blur",
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240816_123832-2csuldv5_epoch6.pt",
        "level": const.TYPE,
    },
    "quality": {
        "asphalt-concrete": {
            "trained_model": "smoothness-asphalt-concrete-efficientNetV2SLinear-20240816_133853-01ci6c3m_epoch14.pt",
            "level": const.QUALITY,
        },
        "paving_stones-sett": {
            "trained_model": "smoothness-paving_stones-sett-efficientNetV2SLinear-20240816_142638-oy2ftm4s_epoch17.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240816_150122-7io2xyxn_epoch15.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk = {
    "model_name": "RTK_complete_lmh-crop",
    "42": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240814_175945-46npupmv_epoch5.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240814_182112-ur7qij9l_epoch19.pt",
                "level": const.QUALITY,
            },
            const.PAVED: {
                "trained_model": "smoothness-paved-efficientNetV2SLinear-20240814_184123-f22wvik6_epoch18.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240814_185347-y5erwqth_epoch15.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1024": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_114256-kfe5xtd5_epoch17.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_121922-zoafkp28_epoch16.pt",
                "level": const.QUALITY,
            },
            const.PAVED: {
                "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_124021-b7i0agdq_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_125320-b6bjvt9w_epoch19.pt",
                "level": const.QUALITY,
            },
        },
    },
    "3": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_162735-qdl2ksr0_epoch7.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_165315-v04esfdf_epoch17.pt",
                "level": const.QUALITY,
            },
            const.PAVED: {
                "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_171540-4huoj27t_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_172826-wrz00mub_epoch16.pt",
                "level": const.QUALITY,
            },
        },
    },
    "57": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_173430-sueob55h_epoch14.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_180938-tf0mk5v4_epoch16.pt",
                "level": const.QUALITY,
            },
            const.PAVED: {
                "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_183043-0nf8w9lm_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_184653-6q3qcugq_epoch9.pt",
                "level": const.QUALITY,
            },
        },
    },
    "1000": {
        "type": {
            "trained_model": "surface-efficientNetV2SLinear-20240820_185210-vho7eou4_epoch17.pt",
            "level": const.TYPE,
        },
        "quality": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_193110-j4iyswqx_epoch17.pt",
                "level": const.QUALITY,
            },
            const.PAVED: {
                "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_195722-txy2gnn2_epoch19.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_201048-u22o0li3_epoch13.pt",
                "level": const.QUALITY,
            },
        },
    },


}

model_dict_rtk_high_blur_42 = {
    "model_name": "RTK_complete_lmh-crop_high_blur_42",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240819_115117-i8h19nhx_epoch14.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240819_123747-94evls58_epoch19.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240819_130604-ywkh0ofc_epoch19.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240819_132134-5zurhyek_epoch19.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk_high_blur_1024 = {
    "model_name": "RTK_complete_lmh-crop_high_blur_1024",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240820_013430-gq0535u7_epoch7.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_020620-hr8lyx02_epoch18.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_023329-cw407xn9_epoch16.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_024932-u63116th_epoch13.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk_high_blur_3 = {
    "model_name": "RTK_complete_lmh-crop_high_blur_3",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240820_025714-8k0w3oa3_epoch3.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_032005-d42jruoy_epoch19.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_034638-vtub1ntp_epoch19.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_040231-fk3hu21h_epoch8.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk_low_blur_42 = {
    "model_name": "RTK_complete_lmh-crop_low_blur_42",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240820_040812-kijdtund_epoch6.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_043731-taldpzcm_epoch17.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_050313-6c8vgvnx_epoch19.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_051907-xzce6ffg_epoch19.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk_low_blur_1024 = {
    "model_name": "RTK_complete_lmh-crop_low_blur_1024",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240820_052629-q709kce6_epoch8.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_055955-x8qs48k9_epoch19.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_062550-ruw03dzf_epoch17.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_064128-19ib6vg8_epoch16.pt",
            "level": const.QUALITY,
        },
    },
}

model_dict_rtk_low_blur_3 = {
    "model_name": "RTK_complete_lmh-crop_low_blur_3",
    # crop lower middle half
    "type": {
        "trained_model": "surface-efficientNetV2SLinear-20240820_064842-7dc9j5xc_epoch4.pt",
        "level": const.TYPE,
    },
    "quality": {
        const.ASPHALT: {
            # crop lower middle half
            "trained_model": "smoothness-asphalt-efficientNetV2SLinear-20240820_071226-atbmtgas_epoch11.pt",
            "level": const.QUALITY,
        },
        const.PAVED: {
            # crop lower middle half
            "trained_model": "smoothness-paved-efficientNetV2SLinear-20240820_073528-7l1iqnz2_epoch19.pt",
            "level": const.QUALITY,
        },
        const.UNPAVED: {
            # crop lower middle half
            "trained_model": "smoothness-unpaved-efficientNetV2SLinear-20240820_075040-01b1au8k_epoch19.pt",
            "level": const.QUALITY,
        },
    },
}

surface_types = {
    "V1_0/test":{
        "V1_0_annotated": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.CONCRETE,
            const.PAVING_STONES: const.PAVING_STONES,
            const.SETT: const.SETT,
            const.UNPAVED: const.UNPAVED,
        },
    },
    "V1_0/downsampled_rtk": {
        "RTK_complete_lmh-crop": {
            "asphalt-concrete": const.ASPHALT,
            "paving_stones-sett": const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
    },
    "V1_0/annotated": {
        "V1_0_merged_lmh-crop": {
            const.ASPHALT: "asphalt-concrete",
            const.CONCRETE: "asphalt-concrete",
            const.PAVING_STONES: "paving_stones-sett",
            const.SETT: "paving_stones-sett",
            const.UNPAVED: const.UNPAVED,
        },
        "V1_0_merged_lmh-crop_blur": {
            const.ASPHALT: "asphalt-concrete",
            const.CONCRETE: "asphalt-concrete",
            const.PAVING_STONES: "paving_stones-sett",
            const.SETT: "paving_stones-sett",
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_high_blur_42": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_high_blur_1024": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_high_blur_3": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_low_blur_42": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_low_blur_1024": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_low_blur_3": {
            const.ASPHALT: const.ASPHALT,
            const.CONCRETE: const.ASPHALT,
            const.PAVING_STONES: const.PAVED,
            const.SETT: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
    },
    "RTK/GT": {
        "V1_0_merged_lmh-crop": {
            const.ASPHALT: "asphalt-concrete",
            const.PAVED: "paving_stones-sett",
            const.UNPAVED: const.UNPAVED,
        },
        "V1_0_merged_lmh-crop_blur": {
            const.ASPHALT: "asphalt-concrete",
            const.PAVED: "paving_stones-sett",
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop": {
            const.ASPHALT: const.ASPHALT,
            const.PAVED: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
        "RTK_complete_lmh-crop_blur": {
            const.ASPHALT: const.ASPHALT,
            const.PAVED: const.PAVED,
            const.UNPAVED: const.UNPAVED,
        },
    },
}

###

model_dict = model_dict_rtk
dataset = dataset_V1_0_downsampled_rtk

for seed in [
    "42",
    "1024",
    "3",
    "57",
    "1000",
    ]:
    config_type = {
        **base_config,
        "name": f"{model_dict["model_name"]}_s{seed}_type_prediction",
        "dataset": f"{dataset}",
        "model_dict": model_dict[seed]["type"]
    } 

    prediction.run_dataset_predict_csv(config_type)

    surfaces = surface_types[dataset][model_dict["model_name"]]
    for surface in surfaces.keys():
        config_quality = {
            **base_config,
            "name": f"{model_dict["model_name"]}_s{seed}_quality_{surface}_prediction",
            "dataset": f"{dataset}/{surface}",
            "model_dict":  model_dict[seed]["quality"][surfaces[surface]], 
        }
        prediction.run_dataset_predict_csv(config_quality)