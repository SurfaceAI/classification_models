import sys
sys.path.append('.')

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os

from experiments.config import global_config

# Bild laden
root_data = global_config.global_config.get("root_data")
dataset = "V12/annotated/paving_stones/good"
image_id = '168730271882986.jpg'
image_path = os.path.join(root_data, dataset, image_id)
image = Image.open(image_path)

# Bild in Tensor konvertieren
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)  # fügt eine Batch-Dimension hinzu

# Funktion zum Anwenden von Gaussian Blur
def apply_gaussian_blur(image_tensor, kernel_size, sigma):
    blurred_image = F.gaussian_blur(image_tensor, kernel_size, sigma=sigma)
    return blurred_image.squeeze(0)  # entfernt die Batch-Dimension

# Verschiedene Werte für Sigma und Kernelgröße
sigma_values = [0.001, 0.1]
sigma_values = [2, 5]
kernel_sizes = [(5, 5), (7, 7), (9, 9), (11, 11)]

# Plotten der originalen und verschwommenen Bilder
fig, axes = plt.subplots(len(sigma_values), len(kernel_sizes) + 1, figsize=(15, 10))

# Originalbild plotten
for i, sigma in enumerate(sigma_values):
    # Originalbild in der ersten Spalte jeder Zeile plotten
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    for j, kernel_size in enumerate(kernel_sizes):
        blurred_image = apply_gaussian_blur(image_tensor, kernel_size, sigma)
        axes[i, j + 1].imshow(transforms.ToPILImage()(blurred_image))
        axes[i, j + 1].set_title(f'Kernel: {kernel_size}, Sigma: {sigma}')
        axes[i, j + 1].axis('off')

plt.tight_layout()
plt.show()
