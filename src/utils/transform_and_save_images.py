import sys
sys.path.append('.')

from PIL import Image
import os
from IPython.display import HTML
from pathlib import Path
from torchvision import transforms


config = {
    'root_path': str(Path().resolve()),
    'root_data': "data/training",
    'dataset': 'V11/annotated/',
    'output_dataset': 'V11/scene_cropped',
    'has_subdirs': True,
}

root_data = os.path.join(config.get('root_path'), config.get('root_data'))

images_path = os.path.join(root_data, config.get('dataset'))
output_path = os.path.join(root_data, config.get('output_dataset'))
has_subdirs = config.get('has_subdirs', True)

def load_process_save_images(images_path, output_path, has_subdirs=True):
    for root, _, fnames in sorted(os.walk(images_path, followlinks=True)):
        for fname in sorted(fnames):
            if fname.endswith((".jpg", ".jpeg", ".png")):
                old_path = os.path.join(root, fname)
                img = Image.open(old_path)
                img_cropped = custom_crop(img, crop_style="lower_half")
                if has_subdirs:
                    subdir_path = os.path.relpath(old_path, images_path)
                    new_path = os.path.join(output_path, subdir_path)
                    if not os.path.exists(os.path.split(new_path)[0]):
                        os.makedirs(os.path.split(new_path)[0])
                    img_cropped.save(new_path)
                else:
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    img_cropped.save(os.path.join(output_path, fname))

def custom_crop(img, crop_style="lower_middle_third"):
    im_width, im_height = img.size
    if crop_style == "lower_middle_third":
        top = im_height / 3 * 2
        left = im_width / 3
        height = im_height - top
        width = im_width / 3
    elif crop_style == "lower_middle_half":
        top = im_height / 2
        left = im_width / 4
        height = im_height / 2
        width = im_width / 2
    elif crop_style == "lower_half":
        top = im_height / 2
        left = 0
        height = im_height / 2
        width = im_width
    else:  # None, or not valid
        return img
    cropped_img = transforms.functional.crop(img, top, left, height, width)
    return cropped_img

load_process_save_images(images_path, output_path, has_subdirs)

