
import sys
sys.path.append('.')

import os
import sys


from experiments.config import predict_config
from src.utils import preprocessing
from src import constants

from IPython.display import Image as IPImage
from IPython.display import display, HTML

import pickle 
from collections import OrderedDict

import time
import pandas as pd
import numpy as np 

from PIL import Image
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
import torch 

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



config = predict_config.B_CNN


features_save_name = config.get("name") + '-' + config.get("dataset").replace('/', '_') + '-features'


# How to load images & embeddings from disc
with open(os.path.join(config.get('evaluation_path'), features_save_name), "rb") as f_in:
    stored_data = pickle.load(f_in)
    stored_ids = stored_data['image_ids']
    stored_coarse_features = stored_data['coarse_features']
    stored_fine_features = stored_data['fine_features']
    stored_predictions = stored_data['prediction']
    

all_labels = pd.read_csv(os.path.join(config.get('data_path'), f'metadata/annotations_combined.csv'), usecols=['image_id', 'surface', 'smoothness'])
true_labels = df[df['image_id'].isin(combined_df['image_id'])]

    
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=config.get('seed')).fit_transform(stored_coarse_features)

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
 
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)



# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)
 
# for every class, we'll add a scatter plot separately
for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]
 
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
 
    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255
 
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)
 
# build a legend using the labels we set previously
ax.legend(loc='best')
 
# finally, show the plot
plt.show()