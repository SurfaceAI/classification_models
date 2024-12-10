import os
import sys

sys.path.append(".")

import argparse
from huggingface_hub import login, HfApi

from deployment.config import model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model upload to huggingface")

    parser.add_argument("--config", type=str, required=True, help="Model config from model_config.py")
    args = parser.parse_args()

    if hasattr(model_config, args.config):
        model_params = getattr(model_config, args.config)
        if isinstance(model_params, dict):  # Optional: Nur Dicts erlauben
            print(f"Login to huggingface...")
            print(model_params.get("hf_token_file"))
            with open(model_params.get("hf_token_file"), "r") as file:
                hf_token = file.read().strip()
            login(token=hf_token)

            print(f"Upload models to huggingface with config parameters '{args.config}'...")
            model_root = model_params.get("model_root")
            model_version = model_params.get("model_version")
            model_repo = model_params.get("hf_model_repo")
            
            # TODO: upload documentation based on model parameters

            api = HfApi()
            api.upload_folder(
                folder_path=os.path.join(model_root, model_version),
                path_in_repo=model_version,
                repo_id=model_repo,
                repo_type="model",
            )

            print("Done.")

        else:
            print(f"'{args.config}' is no dict.")
    else:
        print(f"'{args.config}' does not exist in config.py.")


# import os
# import sys

# from pathlib import Path
# import torch
# import torch.nn as nn
# from torch import Tensor
# from torchvision import models
# from huggingface_hub import PyTorchModelHubMixin

# # local modules
# src_dir = Path(os.path.abspath(__file__)).parent.parent
# sys.path.append(str(src_dir))
# # import utils

# import constants as const

# class CustomEfficientNetV2SLinear(
#     nn.Module,
#     PyTorchModelHubMixin, 
#     # optionally, you can add metadata which gets pushed to the model card
#     repo_url='https://huggingface.co/dthh/test_models',
# ):
#     def __init__(self, num_classes, avg_pool=1):
#         super(CustomEfficientNetV2SLinear, self).__init__()

#         model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
#         # adapt output layer
#         in_features = model.classifier[-1].in_features * (avg_pool * avg_pool)
#         fc = nn.Linear(in_features, num_classes, bias=True)
#         model.classifier[-1] = fc
        
#         self.features = model.features
#         self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
#         self.classifier = model.classifier
#         if num_classes == 1:
#             self.criterion = nn.MSELoss
#         else:
#             self.criterion = nn.CrossEntropyLoss
        
#     @ staticmethod
#     def get_class_probabilies(x):
#         return nn.functional.softmax(x, dim=1)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.features(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         x = self.classifier(x)

#         return x
    
#     def get_optimizer_layers(self):
#         return self.classifier

# def load_model(model):
#     model_path = Path("src/models") / model
#     model_state = torch.load(model_path, map_location='cpu')
#     model_cls = model_mapping[model_state['config']['model']]
#     is_regression = model_state['config']["is_regression"]
#     valid_dataset = model_state['dataset']

#     if is_regression:
#         class_to_idx = valid_dataset.get("class_to_idx")
#         classes = {str(i): cls for cls, i in class_to_idx.items()}
#         num_classes = 1
#     else:
#         classes = valid_dataset.get("classes")
#         num_classes = len(classes)
#     model = model_cls(num_classes)
#     model.load_state_dict(model_state['model_state_dict'])

#     return model, classes, is_regression

# model_mapping = {
#     const.EFFNET_LINEAR: CustomEfficientNetV2SLinear,
# }

# model_file_name = "surface-efficientNetV2SLinear-20240923_171219-2t59l5b9_epoch10.pt"
# # create model
# # config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
# # model = MyModel(**config)
# model, _, _ = load_model(model_file_name)

# # save locally
# model.save_pretrained("surface_test")

# # push to the hub
# model.push_to_hub("dthh/surface_test")

# # reload
# model = CustomEfficientNetV2SLinear.from_pretrained("dthh/surface_test")
# print(model.classifier)