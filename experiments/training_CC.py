import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import create_valid_dataset

# for seed in [
#     # 42,
#     # 1024,
#     3,
#     57,
#     1000,
#     ]:
#     config=train_config.train_valid_split_params_rtk.copy()
#     config["root_data"] = config["root_data_rtk"]
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     create_valid_dataset.save_train_valid_split(config=config)

#     config=train_config.efficientnet_surface_params_rtk.copy()
#     config["root_data"] = config["root_data_rtk"]
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     training.run_training(config=config)

#     config=train_config.efficientnet_quality_params_rtk.copy()
#     config["root_data"] = config["root_data_rtk"]
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     training.run_training(config=config)

# for seed in [
#     # 42,
#     # 1024,
#     3,
#     57,
#     1000,
#     ]:
#     config=train_config.train_valid_split_params.copy()
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     create_valid_dataset.save_train_valid_split(config=config)

#     config=train_config.efficientnet_surface_params.copy()
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     training.run_training(config=config)

#     config=train_config.efficientnet_quality_params.copy()
#     config["seed"] = seed
#     config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#     training.run_training(config=config)

for seed in [
    42,
    # 1024,
    # 3,
    # 57,
    # 1000,
    ]:
    config=train_config.train_valid_split_params_V1_0all.copy()
    config["seed"] = seed
    config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
    create_valid_dataset.save_train_valid_split(config=config)

    config=train_config.efficientnet_surface_params_V1_0all.copy()
    config["seed"] = seed
    config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
    training.run_training(config=config)

    # config=train_config.efficientnet_quality_params_V1_0all.copy()
    # config["seed"] = seed
    # config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
    # training.run_training(config=config)

# for kernel, sigma in [[11, 5], [5, 2]]:
#     for seed in [42, 1024, 3]:
#         config=train_config.train_valid_split_params_rtk.copy()
#         config["root_data"] = config["root_data_rtk"]
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         create_valid_dataset.save_train_valid_split(config=config)

#         if (seed == 42) & ([kernel, sigma] == [11, 5]):
#             continue
        
#         config=train_config.efficientnet_surface_params_rtk.copy()
#         config["root_data"] = config["root_data_rtk"]
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         config["augment"]["gaussian_blur_kernel"] = kernel
#         config["augment"]["gaussian_blur_sigma"] = sigma
#         training.run_training(config=config)

#         config=train_config.efficientnet_quality_params_rtk.copy()
#         config["root_data"] = config["root_data_rtk"]
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         config["augment"]["gaussian_blur_kernel"] = kernel
#         config["augment"]["gaussian_blur_sigma"] = sigma
#         training.run_training(config=config)

# for kernel, sigma in [[5, 2], [11, 5]]:
#     for seed in [42, 1024, 3]:
#         config=train_config.train_valid_split_params.copy()
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         create_valid_dataset.save_train_valid_split(config=config)

#         if (seed in [42, 1024]) & ([kernel, sigma] == [11, 5]):
#             continue

#         config=train_config.efficientnet_surface_params.copy()
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         config["augment"]["gaussian_blur_kernel"] = kernel
#         config["augment"]["gaussian_blur_sigma"] = sigma
#         training.run_training(config=config)

#         config=train_config.efficientnet_quality_params.copy()
#         config["seed"] = seed
#         config["train_valid_split_list"] = f"s{seed}_{config["train_valid_split_list"]}"
#         config["augment"]["gaussian_blur_kernel"] = kernel
#         config["augment"]["gaussian_blur_sigma"] = sigma
#         training.run_training(config=config)

