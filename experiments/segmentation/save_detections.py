import sys
sys.path.append('.')

import os
from experiments.config import predict_config
from src.utils import preprocessing, mapillary_requests

config = predict_config.seg_analysis

# image ids
image_data = preprocessing.PredictImageFolder(root=os.path.join(config.get("root_data"), config.get("dataset")))
image_ids = [image_id for _, image_id in image_data.samples]

# load detections
for image_id in image_ids:
    # print(image_id)
    mapillary_requests.save_image_data(
        saving_folder=os.path.join(config.get("root_data"), config.get("segmentation_folder"), 'detections'),
        saving_postfix=config.get('saving_postfix'),
        image_id=image_id,
        access_token=mapillary_requests.load_mapillary_token(
            token_path=os.path.join(config.get("root"), config.get("mapillary_token_file"))
            ),
        url=False,
        detections=True)
    