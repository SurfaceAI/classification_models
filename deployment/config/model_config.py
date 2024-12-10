from experiments.config  import global_config

model_naming = {
    "surface_type": "surface_type",
    "surface_quality": "surface_quality",
    "road_type": "road_type",
}

model_params_v1 = {
    "model_root": global_config.global_config.get("root_model"),
    "hf_model_repo": "SurfaceAI/models",
    "hf_token_file": "huggingface_token.txt",
    "models": {
        "surface_type": "surface-efficientNetV2SLinear-20240923_171219-2t59l5b9_epoch10.pt",
        "surface_quality": {
            "asphalt": "smoothness-asphalt-efficientNetV2SLinear-20240923_144409-86vpv5bs_epoch29.pt",
            "concrete": "smoothness-concrete-efficientNetV2SLinear-20240924_020702-32bp575u_epoch16.pt",
            "paving_stones": "smoothness-paving_stones-efficientNetV2SLinear-20240924_035145-0u6eheod_epoch18.pt",
            "sett": "smoothness-sett-efficientNetV2SLinear-20240924_103221-xrpbxnjc_epoch26.pt",
            "unpaved": "smoothness-sett-efficientNetV2SLinear-20240924_103221-xrpbxnjc_epoch26.pt"
        },
        "road_type": "flatten-efficientNetV2SLinear-20240917_125206-9lg7mdeu_epoch10.pt"
    },
    "model_naming": model_naming,
    "model_version": "tbd",
}