import sys
sys.path.append('.')

import os
import shutil

import pandas as pd
from experiments.config import global_config



def create_annotated_image_folders(input_path, output_path, df):
    
    # train_path = os.path.join(output_path, "train")
    # os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join(output_path, "test")
    os.makedirs(test_path, exist_ok=True)
    
    # Iterate through each row in the DataFrame
    for _, row in df[
        df.surface_type.notna() & df.surface_quality.notna()
    ].iterrows():
        if row.train:
            # output_path = train_path
            continue
        else:
            output_path = test_path

        # Create subfolder for surface if not exists
        surface_folder = os.path.join(output_path, row["surface_type"])
        os.makedirs(surface_folder, exist_ok=True)

        # Create subfolder for smoothness if not exists
        smoothness_folder = os.path.join(surface_folder, row["surface_quality"])
        os.makedirs(smoothness_folder, exist_ok=True)

        # Copy the image to the respective folder
        destination_path = os.path.join(smoothness_folder, f"{row['mapillary_image_id']}.jpg")
        image_filename = os.path.join(input_path, f"{row['mapillary_image_id']}.jpg")
        if os.path.exists(image_filename):
            shutil.copy(image_filename, destination_path)


if __name__ == "__main__":

    root_data = global_config.global_config.get("root_data")
    dataset = "V1_0"
    input_folder = "s_1024"
    output_folder = ""
    metadata_folder = "metadata"
    metadata_file_name = "streetSurfaceVis_v1_0.csv"

    annotations = pd.read_csv(
                    os.path.join(root_data, dataset, metadata_folder, metadata_file_name),
                    dtype={"mapillary_image_id": str,
                        "train": bool},
                    index_col=False,
                )
    input_path = os.path.join(root_data, dataset, input_folder)
    output_path = os.path.join(root_data, dataset, output_folder)

    create_annotated_image_folders(input_path=input_path, output_path=output_path, df=annotations)
