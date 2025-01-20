import sys

sys.path.append(".")

import argparse
from huggingface_hub import login, HfApi

from deployment.config import model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code upload to huggingface")

    parser.add_argument("-c", "--config", type=str, required=True, help="Repo config from model_config.py")
    args = parser.parse_args()

    if hasattr(model_config, args.config):
        repo_params = getattr(model_config, args.config)
        if isinstance(repo_params, dict):
            print(f"Login to huggingface...")
            print(repo_params.get("hf_token_file"))
            with open(repo_params.get("hf_token_file"), "r") as file:
                hf_token = file.read().strip()
            login(token=hf_token)

            print(f"Upload code to huggingface...")
            local_root = repo_params.get("local_root") / "deployment" / "hf_code"
            model_repo = repo_params.get("hf_model_repo")

            api = HfApi()

            # example images folder
            api.upload_folder(
                folder_path=(local_root / "example_images"),
                path_in_repo="example_images",
                repo_id=model_repo,
                repo_type="model",
            )

            # code files
            api.upload_file(
                path_or_fileobj=(local_root / "Models.py"),
                path_in_repo="Models.py",
                repo_id=model_repo,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=(local_root / "prediction_example.py"),
                path_in_repo="prediction_example.py",
                repo_id=model_repo,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=(local_root / "README.md"),
                path_in_repo="README.md",
                repo_id=model_repo,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=(local_root / "requirements.txt"),
                path_in_repo="requirements.txt",
                repo_id=model_repo,
                repo_type="model",
            )


            print("Done.")

        else:
            print(f"'{args.config}' is no dict.")
    else:
        print(f"'{args.config}' does not exist in config.py.")
