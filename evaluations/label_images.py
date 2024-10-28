import sys
sys.path.append('.')

import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

from experiments.config.global_config import ROOT_DIR

data_path = os.path.join(ROOT_DIR, "data", "road_scenery_experiment", "classified_images_add_on_c")
input_dir = os.path.join(data_path, "sidewalk")

# new folder
output_dirs = {
    # 'cycleway (Hochbord)': os.path.join(data_path, "1_2_bicycle", "1_2_cycleway"),
    # 'lane': os.path.join(data_path, "1_2_bicycle", "1_2_lane"),
    'footway': os.path.join(data_path, "1_3_pedestrian", "1_3_footway"),
    'path': os.path.join(data_path, "1_4_path", "1_4_path_unspecified"),
    'no focus': os.path.join(data_path, "2_1_no_focus_no_street", "2_1_all"),
}

for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

output_cat = dict()
text = "Press "
for i, k in enumerate(output_dirs.keys()):
    output_cat[str(i+1)] = k
    text += f"'{i+1}' for {k}, "
text += "'p' for next"

def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def sort_images(input_dir):
    images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in images:
        print(f"Image: {image_file}")
        print(text)

        image_path = os.path.join(input_dir, image_file)
        show_image(image_path)
        
        # Eingabe vom Benutzer
        key = input().lower()

        if key in output_cat.keys():
            shutil.move(image_path, os.path.join(output_dirs[output_cat[key]], image_file))
            print(f"{image_file} -> {key} moved")
        # if key == '1':
        #     shutil.move(image_path, os.path.join(output_dirs['Kategorie1'], image_file))
        #     print(f"{image_file} -> Kategorie1 verschoben")
        # elif key == '2':
        #     shutil.move(image_path, os.path.join(output_dirs['Kategorie2'], image_file))
        #     print(f"{image_file} -> Kategorie2 verschoben")
        # elif key == '3':
        #     shutil.move(image_path, os.path.join(output_dirs['Kategorie3'], image_file))
        #     print(f"{image_file} -> Kategorie3 verschoben")
        elif key == 'p':
            print(f"{image_file} next...")
        else:
            print("invalid key, next...")

if __name__ == "__main__":
    sort_images(input_dir)
