import sys
sys.path.append('.')

from bertopic.representation import KeyBERTInspired, VisualRepresentation
from bertopic.backend import MultiModalBackend
from bertopic import BERTopic
from PIL import Image
import os
import base64
from io import BytesIO
from IPython.display import HTML
from pathlib import Path
from torchvision import transforms
from datetime import datetime
import pandas as pd
import numpy as np
import time
import torch
import pickle
from umap import UMAP
from hdbscan import HDBSCAN

config = {
    'embedding_model': MultiModalBackend('clip-ViT-B-32', batch_size=32),
    'representation_model': {
        "Visual_Aspect": VisualRepresentation(image_to_text_model="nlpconnect/vit-gpt2-image-captioning")
        },
    'root_path': str(Path().resolve()),
    'root_data': "data/training",
    'saving_root_path': "trained_models",
    'name': 'bertopic',
    'dataset': 'V11/scene_cropped/',
    'metadata_path': 'V11/metadata',
    'metadata_file': 'annotations_combined.csv',
    'embeddings_path': None,
    'embeddings_path': "embeddings/img_cropped_scene_embeddings_new_path.pkl",
    'embeddings_path': "embeddings/effnet_feature_5_embeddings.pkl",
    'embeddings_path': "embeddings/efficientnet_v2_s_feature_embeddings.pkl",
    'min_topic_size': 10,
    'hdbscan_metric': 'euclidean', # no 'cosine' implemented!
    'cluster_selection': 'leaf', # 'leaf, 'eom'
    'random_state': 42,
    'n_components': 5,
    'n_neighbors': 15,
    # 'nr_topics': 'auto',
}

# check possible hdbsacn_metric with:
# from sklearn import neighbors
# neighbors.BallTree.valid_metrics

# # Image embedding model
# embedding_model = config.get('embedding_model')

# # Image to text representation model
# representation_model = config.get('representation_model')

# images
root_path = config.get('root_path')
root_data = os.path.join(root_path, config.get('root_data'))
saving_root_path = os.path.join(root_path, config.get('saving_root_path'))
name = config.get("name", "")

start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
saving_name = name + "-" + start_time
saving_path = os.path.join(saving_root_path, saving_name)

if config.get('embeddings_path') is not None:
    with open(os.path.join(root_path, config.get('embeddings_path')), "rb") as f_in:
        data = pickle.load(f_in)
        images = data['images']
        embeddings = data['embeddings'].cpu().numpy()
        
else:
    # images_path = os.path.join(root_data, 'V9/annotated/asphalt/bad')
    # images_path = os.path.join(root_data, 'seg_analysis/original/not_recognizable/multi/streets')
    images_path = os.path.join(root_data, config.get('dataset'))
    embeddings = None
    images = []
    for root, _, fnames in sorted(os.walk(images_path, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if fname.endswith((".jpg", ".jpeg", ".png")):
                images.append(path)

# Image embedding model
embedding_model = config.get('embedding_model')

# Image to text representation model
representation_model = config.get('representation_model')

# umap
umap_model = UMAP(n_neighbors=config.get('n_neighbors', 15),
                  n_components=config.get('n_components', 5),
                  min_dist=0.0,
                  metric='cosine',
                  low_memory=False,
                  random_state=config.get('random_state', 42),
                  )

# hdbscan
hdbscan_model = HDBSCAN(min_cluster_size=config.get('min_topic_size', 10),
                        metric=config.get('hdbscan_metric', 'euclidean'),
                        cluster_selection_method=config.get('cluster_selection'),
                        prediction_data=True)

# for guided topic modeling only
seed_topic_list = [["bikelane on road"],
                   ["bikelane on sidewalk"],
                   ["driving"]]
# for zero-shot modeling only
zeroshot_topic_list = ["bikelane on road", "bikelane on sidewalk"]
# for semi-supervised modeling only
# partial_labels = {'115613061245645': 'bikelane on sidewalk',
#                   '147530060597553': 'biklane on street',
#                   '152436594516951': 'bikelane on sidewalk',
#                   '161779362788691': 'bikelane on sidewalk',
#                   '314600876906279': 'biklane on street',
#                   '324264732453909': 'bikelane on sidewalk',
#                   '332779451733060': 'bikelane on sidewalk',
#                   '371383354168821': 'biklane on street',
#                   '406638231063174': 'bikelane on sidewalk',
#                   '476023756839183': 'bikelane on sidewalk',
#                   '485652966084685': 'bikelane on sidewalk',
#                   '527494424951466': 'biklane on street',
#                   '586506369409581': 'biklane on street',
#                   '645723314281364': 'biklane on street',
#                   }
# labels_categories = ['bikelane on sidewalk', 'biklane on street']
# def labels_index(label):
#     class_to_idx = {labels_categories[i]: i for i in range(len(labels_categories))}
#     return -1 if label not in class_to_idx.keys() else class_to_idx[label]
# ids = [os.path.splitext(os.path.split(path)[-1])[0] for path in images]
# labels = [partial_labels.get(id, 'no label') for id in ids]
# y = [labels_index(label) for label in labels]
# load annotation data
df = pd.read_csv(os.path.join(root_data, config.get('metadata_path'), config.get('metadata_file')))
labels_subset = np.unique(df['roadtype'].apply(str))
df['roadtype'] = df['roadtype'].fillna('no_roadtype')
df['image_id'] = df['image_id'].apply(str)
roadtypes = []
for image_path in images:
    id = os.path.splitext(os.path.split(image_path)[-1])[0]
    # TODO: default if roadtype not set
    roadtypes.append(df[df['image_id'] == id].loc[:,'roadtype'].iloc[0])
roadtyp_classes = [s for s in list(set(roadtypes)) if s != 'no_roadtype']
def labels_index(label):
    class_to_idx = {roadtyp_class: i for i, roadtyp_class in enumerate(roadtyp_classes)}
    return -1 if label not in class_to_idx.keys() else class_to_idx[label]
y = list(map(labels_index, roadtypes))


# transform images
# images = [Image.open(img) for img in images]
# images = [transforms.functional.crop(img, 0.5*img.size[1], 0, 0.5*img.size[1], img.size[0]) for img in images]

# standard
# topic_model = BERTopic(embedding_model=embedding_model,
#                        representation_model=representation_model,
#                        umap_model=umap_model,
#                        hdbscan_model=hdbscan_model,
#                        nr_topics=config.get('nr_topics', None),
#                     #    min_topic_size=config.get('min_topic_size', 10), # included in hdbscan
#                        )
# topics, probs = topic_model.fit_transform(documents=None, images=images, embeddings=embeddings)
# # guided topic modeling
# topic_model = BERTopic(embedding_model=embedding_model,
#                        representation_model=representation_model,
#                        min_topic_size=5,
#                        seed_topic_list=seed_topic_list)
# topics, probs = topic_model.fit_transform(documents=None, images=images)
# # zero-shot topic modeling
# topic_model = BERTopic(embedding_model=embedding_model,
#                        representation_model=representation_model,
#                     #    min_topic_size=5,
#                        zeroshot_topic_list=zeroshot_topic_list,
#                        zeroshot_min_similarity=0.7,)
# topics, probs = topic_model.fit_transform(documents=None, images=images)
# # semi-supervised topic modeling
topic_model = BERTopic(embedding_model=embedding_model,
                       representation_model=representation_model,
                       umap_model=umap_model,
                       hdbscan_model=hdbscan_model,
                       nr_topics=config.get('nr_topics', None),
                    #    min_topic_size=config.get('min_topic_size', 10), # included in hdbscan
                       )
topics, probs = topic_model.fit_transform(documents=None,
                                          images=images,
                                          embeddings=embeddings,
                                          y=y)



def image_base64(im):
    if isinstance(im, str):
        # im = get_thumbnail(im)
        print("is string")
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

print(topic_model.get_topic_info())

# topic_model.save(path=saving_path)
# torch.save(topic_model, saving_path)
# safetensors saves several files in a folder
topic_model.save(path=saving_path, serialization="safetensors", save_ctfidf=True)
print(f'topic model saved: {saving_path}.')

# save csv with information, error for nr_topics != None
if not config.get('nr_topics', None):
    info = topic_model.get_document_info(images)
    saving_name = name + "-" + start_time + ".csv"
    saving_path = os.path.join(saving_root_path, saving_name)
    info.to_csv(saving_path, index=False)
    print(f'topic model info saved: {saving_path}.')


print('Done.')

df = topic_model.get_topic_info().drop("Representative_Docs", axis=1).drop("Name", axis=1)

# Visualize the images
HTML(df.to_html(formatters={'Visual_Aspect': image_formatter}, escape=False))

