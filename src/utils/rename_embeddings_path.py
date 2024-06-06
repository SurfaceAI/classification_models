import sys
sys.path.append('.')

import os
import pickle
from pathlib import Path

config = {
    'root_path': str(Path().resolve()),
    'folder': "embeddings",
    'file_old': 'img_cropped_scene_embeddings.pkl',
    'file_new': 'img_cropped_scene_embeddings_new_path.pkl',
}

OLD_STORAGE = '/storage'
NEW_STORAGE = '/home'

with open(os.path.join(config.get('root_path'), config.get('folder'), config.get('file_old')), "rb") as f_in:
    stored_data = pickle.load(f_in)
    stored_images = stored_data['images']


new_stored_images = []
for old_path in stored_images:
    sub_path = os.path.relpath(old_path, OLD_STORAGE)
    new_path = os.path.join(NEW_STORAGE, sub_path)
    new_stored_images.append(new_path)
stored_data['images'] = new_stored_images

with open(os.path.join(config.get('root_path'), config.get('folder'), config.get('file_new')), "wb") as f_out:
    pickle.dump(stored_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)
