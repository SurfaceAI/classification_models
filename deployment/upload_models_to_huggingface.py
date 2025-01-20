import os
import sys

sys.path.append(".")

import argparse
from huggingface_hub import login, HfApi

from deployment.config import model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model upload to huggingface")

    parser.add_argument("-c", "--config", type=str, required=True, help="Model config from model_config.py")
    args = parser.parse_args()

    if hasattr(model_config, args.config):
        model_params = getattr(model_config, args.config)
        if isinstance(model_params, dict):
            print(f"Login to huggingface...")
            print(model_params.get("hf_token_file"))
            with open(model_params.get("hf_token_file"), "r") as file:
                hf_token = file.read().strip()
            login(token=hf_token)

            print(f"Upload models to huggingface with config parameters '{args.config}'...")
            model_root = model_params.get("model_root")
            model_version = model_params.get("model_version")
            model_repo = model_params.get("hf_model_repo")
            

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
