import sys
sys.path.append('.')

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pickle
import torch
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import LabelEncoder

from src.models import prediction
from src.utils import helper

# embeddings
# embeddings_saved = 'img_cropped_scene_embeddings_new_path.pkl'
# embeddings_saved = 'dino_embeddings.pkl'
embeddings_saved = 'effnet_feature_5_embeddings.pkl'
# embeddings_saved = 'efficientnet_v2_s_feature_embeddings.pkl'

# visualization method
visualization = 'tsne' # 'umap', 'tsne'



root_path = str(Path().resolve())
embeddings_folder = 'embeddings'
data_path = 'data/training'
dataset = 'V11/scene_cropped'
metadata_path = 'V11/metadata'
metadata_file = 'annotations_combined.csv'
saving_folder = 'visualizations'
seed = 42
gpu_kernel = 0

viz_name = ''
# viz_name = 'all_labels'

# load device
device = torch.device(
    f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
)

# load embeddings
p = os.path.join(root_path, embeddings_folder, embeddings_saved)
with open(os.path.join(root_path, embeddings_folder, embeddings_saved), "rb") as f_in:
    # data = torch.load(f_in, map_location=device) # error: magic_number invalid on gpu device, error on cpu device
    data = pickle.load(f_in)
    images = data['images']
    embeddings = data['embeddings']

embeddings = embeddings.cpu().numpy()


# load annotation data
df = pd.read_csv(os.path.join(root_path, data_path, metadata_path, metadata_file))
roadtypes_subset = np.unique(df['roadtype'].apply(str))
surface_subset = np.unique(df['surface'].apply(str))
labels_subset = [f'{s}_{r}' for s in surface_subset for r in roadtypes_subset]
df['roadtype'] = df['roadtype'].fillna('no_roadtype').apply(str)
df['surface'] = df['surface'].fillna('no_surface').apply(str)
df['label'] = df['surface'] + '_' + df['roadtype']
df['image_id'] = df['image_id'].apply(str)
labels = []
for image_path in images:
    id = os.path.splitext(os.path.split(image_path)[-1])[0]
    # TODO: default if roadtype not set
    labels.append(df[df['image_id'] == id].loc[:,'label'].iloc[0])
roadtypes_labels = []
for image_path in images:
    id = os.path.splitext(os.path.split(image_path)[-1])[0]
    # TODO: default if roadtype not set
    roadtypes_labels.append(df[df['image_id'] == id].loc[:,'roadtype'].iloc[0])


# dim reduction with t-sne or umap

if visualization == 'umap':
    reduced_embeddings = UMAP(n_neighbors=15,
                                    n_components=2,
                                    min_dist=0.0,
                                    metric='cosine',
                                    low_memory=False,
                                    random_state=seed,
                                    ).fit_transform(embeddings)
elif visualization == 'tsne':
    reduced_embeddings = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, random_state=seed).fit_transform(embeddings)
else:
    print('No method chosen for visualization!')

# sns.set_theme(style="white", context="poster", rc={ "figure.figsize": (14, 10), "lines.markersize": 1 })

# def make_scatter(x, y=None):
#     if y is None:
#         plt.scatter(x[idx, 0], x[idx, 1])
#     else:
#         for label in np.unique(y):
#             idx = np.where(np.array(y) == label)[0]
#             plt.scatter(x[idx, 0], x[idx, 1], label=label)
#             plt.legend(bbox_to_anchor=(0, 1), loc="upper left", markerscale=6)

# def plot(x, filename, y=None):
#     make_scatter(x, y)
#     # plt.xlabel("UMAP 1")
#     # plt.ylabel("UMAP 2")
#     plt.title(filename)
#     if not os.path.exists(os.path.join(root_path, saving_folder)):
#         os.mkdir(os.path.join(root_path, saving_folder))
#     plt.savefig(os.path.join(root_path, saving_folder, filename))
#     # plt.show()

def generate_color_palette(num_colors):
    # Generate a set of distinguishable colors
    colors = sns.color_palette("hsv", num_colors)
    return colors

def create_plot(tsne_data, tsne_label, save_name, labels_subset=None):
    
    label_encoder = LabelEncoder()
    scatter_labels_encoded = label_encoder.fit_transform(tsne_label)
    
    if labels_subset is not None:
        subset_indices = np.isin(tsne_label, labels_subset)
        tsne_data = tsne_data[subset_indices]
        scatter_labels_encoded = scatter_labels_encoded[subset_indices]
    
    num_labels = len(np.unique(scatter_labels_encoded))
    
    # Generate distinguishable colors
    colors = generate_color_palette(num_labels)

    # Create a scatter plot
    plt.figure(figsize=(20, 16))
    for i, label in enumerate(np.unique(scatter_labels_encoded)):
        indices = np.where(scatter_labels_encoded == label)
        plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1], c=[colors[i]], label=label_encoder.classes_[label], s=10)

    plt.title(f'{save_name}_{os.path.splitext(embeddings_saved)[0]}_{visualization}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    if not os.path.exists(os.path.join(root_path, saving_folder)):
        os.mkdir(os.path.join(root_path, saving_folder))
    plt.savefig(os.path.join(root_path, saving_folder, f'_{save_name}_{os.path.splitext(embeddings_saved)[0]}_{visualization}_plot.jpeg'))

# visualization
# plot(reduced_embeddings, '', y=roadtypes)
create_plot(reduced_embeddings, roadtypes_labels, viz_name, labels_subset=roadtypes_subset)

# color for roadtype

