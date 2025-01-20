import os
from pathlib import Path
from PIL import Image
import logging

import Models

config = {
    "model_root": "models",
    "hf_model_repo": "SurfaceAI/models",
    "models": {
        "surface_type": "v1/surface_type_v1.pt",
        "surface_quality": {
            "asphalt": "v1/surface_quality_asphalt_v1.pt",
            "concrete": "v1/surface_quality_concrete_v1.pt",
            "paving_stones": "v1/surface_quality_paving_stones_v1.pt",
            "sett": "v1/surface_quality_sett_v1.pt",
            "unpaved": "v1/surface_quality_unpaved_v1.pt"
        },
        "road_type": "v1/road_type_v1.pt"
    },
    "gpu_kernel": 0,
    "transform_surface": {
        "resize": 384,
        "crop": "lower_middle_half"
    },
    "transform_road_type": {
        "resize": 384,
        "crop": "lower_half"
    },
}

root_path = Path(os.path.abspath(__file__)).parent

image_ids = [
    # "IMG_20210221_135447",
    "IMG_20210226_172956",
    # "IMG_20230130_162826",
]

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

image_data = []
for id in image_ids:
    path = root_path / "example_images" / f"{id}.jpg"
    try:
        image_data.append(Image.open(path))
    except Exception as e:
        logging.warning(f'{e}: Not found or corrupted image: {path}')

md = Models.ModelInterface(config=config)
results = md.batch_classifications(image_data, image_ids)
for result in results:
    print(result)