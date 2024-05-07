# %%
#Imports
import sys
sys.path.append('.')
sys.path.append('..')


import os
from experiments.config import predict_config

import pickle 
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# %%
config = predict_config.B_CNN


# %%
#Load feature vecotrs
features_save_name = 'multi_label_prediction-V11_annotated-features'
predictions_save_name = 'multi_label_prediction-V11_annotated-20240507_155324.csv'


with open(os.path.join(config.get('evaluation_path'), features_save_name), "rb") as f_in:
    stored_data = pickle.load(f_in)
    stored_ids = stored_data['image_ids']
    stored_coarse_features = stored_data['coarse_features']
    stored_fine_features = stored_data['fine_features']
    stored_predictions = stored_data['prediction']


# %%
stored_df = pd.DataFrame({'image_id': stored_ids, 'coarse_features': str(stored_coarse_features),
                          'fine_features': str(stored_fine_features)})

# %%
all_encodings = []  # Initialize an empty DataFrame
index = 0
for id in stored_ids:
    coarse_feat = stored_coarse_features[index]
    fine_feat = stored_fine_features[index]
    
    data = {'image_id': int(id), 'fine_feat': str(fine_feat), 'coarse_feat': str(coarse_feat)}
    #row_df = pd.DataFrame(data, index=[index])  # Create a DataFrame from the dictionary
    all_encodings.append(data)  # Append the row DataFrame to the main DataFrame
    index += 1
    
stored_df = pd.DataFrame(all_encodings)    

    

# %%
#load the true labels
all_labels = pd.read_csv(os.path.join(config.get('root_data'), f'V11/metadata/annotations_combined.csv'), usecols=['image_id', 'surface', 'smoothness'])
all_labels = all_labels[~all_labels['smoothness'].isna()]
all_labels = all_labels[~all_labels['surface'].isna()]



# %%
#adding true labels to our stored_df

stored_df = pd.merge(stored_df, all_labels, how="left", left_on="image_id",
                     right_on="image_id")


# %%
#separating our stored_df in valid and training data
all_predictions = pd.read_csv(os.path.join(config.get('root_predict'), predictions_save_name))
all_predictions = all_predictions.rename(columns = {"Image":"image_id"})
all_predictions['image_id'] = all_predictions['image_id'].astype('int64')
valid_predictions = all_predictions[all_predictions['is_in_validation'] == 0]
train_predictions = all_predictions[all_predictions['is_in_validation'] == 1]

all_predictions



# %%
# merge all_predictions with stored_df
valid_df = pd.merge(stored_df, all_predictions[all_predictions['is_in_validation'] == 1],
                     how='inner', on='image_id')

train_df = pd.merge(stored_df, all_predictions[all_predictions['is_in_validation'] == 0],
                     how='inner', on='image_id')

# %%
id_position = {image_id: position for position, image_id in enumerate(stored_ids)}
valid_df['image_id'] = valid_df['image_id'].astype('str')
valid_df['position'] = valid_df['image_id'].map(id_position)
train_df['image_id'] = train_df['image_id'].astype('str')
train_df['position'] = train_df['image_id'].map(id_position)


# %%
validation_input_coarse_tsne = stored_coarse_features[valid_df['position'].to_list()]
validation_labels_coarse_tsne = valid_df['surface'].to_list()

train_input_coarse_tsne = stored_coarse_features[train_df['position'].to_list()]
train_labels_coarse_tsne = train_df['surface'].to_list()

validation_input_fine_tsne = stored_fine_features[valid_df['position'].to_list()]
validation_labels_fine_tsne = valid_df['smoothness'].to_list()

train_input_fine_tsne = stored_fine_features[train_df['position'].to_list()]
train_labels_fine_tsne = train_df['smoothness'].to_list()




# %%


tsne_coarse_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=config.get('seed')).fit_transform(validation_input_coarse_tsne)
tsne_coarse_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=config.get('seed')).fit_transform(train_input_coarse_tsne)

tsne_fine_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=config.get('seed')).fit_transform(validation_input_fine_tsne)
tsne_fine_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=config.get('seed')).fit_transform(train_input_fine_tsne)


# %%
from sklearn.preprocessing import LabelEncoder

def create_plot(tsne_data, tsne_label, flag):
    label_encoder = LabelEncoder()
    scatter_labels_encoded = label_encoder.fit_transform(tsne_label)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=scatter_labels_encoded, cmap='viridis', s=10)
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Surface Type').set_ticklabels(label_encoder.classes_)
    plt.title('t-SNE coarse features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(config.get('evaluation_path'), f'{flag}_tsne_plot_validation.jpeg'))

# %%
create_plot(tsne_coarse_train, train_labels_coarse_tsne, 'train_coarse')
create_plot(tsne_coarse_valid, validation_labels_coarse_tsne, 'valid_coarse')

create_plot(tsne_fine_train, train_labels_fine_tsne, 'train_fine')
create_plot(tsne_fine_valid, validation_labels_fine_tsne, 'valid_fine')




# %%



