import os
import sys

sys.path.append(".")

import torch
import argparse

from deployment.config import model_config
from src.utils import helper


def save_model(model_root, model_old, model_new, validation=False):
    model_path = os.path.join(model_root, model_old)
    model_state = torch.load(model_path, map_location="cpu")
    model_name = model_state["config"]["model"]
    is_regression = model_state["config"]["is_regression"]
    valid_dataset = model_state["dataset"]

    class_to_idx = valid_dataset.get("class_to_idx")

    model_state_dict = model_state["model_state_dict"]

    data = {
        "model_state_dict": model_state_dict,
        "is_regression": is_regression,
        "class_to_idx": class_to_idx,
        "model_name": model_name,
    }

    model_path = os.path.join(model_root, model_new)
    directory = os.path.dirname(model_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(data, model_path)

    # for validation only
    if validation:
        model_state = torch.load(model_path, map_location="cpu")
        model_name = model_state["model_name"]
        is_regression = model_state["is_regression"]
        class_to_idx = model_state["class_to_idx"]
        # idx_to_class = {str(i): cls for cls, i in class_to_idx.items()}
        num_classes = 1 if is_regression else len(class_to_idx.items())
        model_state_dict = model_state["model_state_dict"]
        model_cls = helper.string_to_object(model_name)
        model = model_cls(num_classes)
        model.load_state_dict(model_state_dict)
        print(f"Successful loading of the converted model {model_new}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model conversion")

    parser.add_argument("--config", type=str, required=True, help="Model config from model_config.py")
    parser.add_argument("--validation", type=str, required=False, help="if 'y', then model loading is done for validation of conversion.")
    args = parser.parse_args()

    if hasattr(model_config, args.config):
        model_params = getattr(model_config, args.config)
        if isinstance(model_params, dict):  # Optional: Nur Dicts erlauben
            print(f"Convert models with config parameters '{args.config}'...")

            model_root = model_params.get("model_root")

            model_files_old = model_params.get("models")
            model_files_new = model_params.get("model_naming")
            model_version = model_params.get("model_version")

            # TODO: write documentation based on model parameters

            validation = True if args.validation.lower() == "y" else False

            # surface model
            save_model(
                model_root,
                model_files_old["surface_type"],
                os.path.join(model_version, "_".join([model_files_new["surface_type"], model_version]) + ".pt"),
                validation,
            )

            # quality models
            for tp, model in model_files_old["surface_quality"].items():
                save_model(
                    model_root,
                    model,
                    os.path.join(model_version, "_".join([model_files_new["surface_quality"], tp, model_version]) + ".pt"),
                    validation,
                )

            # road type model
            save_model(
                model_root,
                model_files_old["road_type"],
                os.path.join(model_version, "_".join([model_files_new["road_type"], model_version]) + ".pt"),
                validation,
            )

            print("Done.")

        else:
            print(f"'{args.config}' is no dict.")
    else:
        print(f"'{args.config}' does not exist in config.py.")

