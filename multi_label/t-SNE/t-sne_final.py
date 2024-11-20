# %%
#Imports
import sys
sys.path.append('.')
sys.path.append('..')
import os
import pickle 
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# %%
#config = predict_config.B_CNN
seed=42
evaluation_path = r"C:\Users\esthe\Documents\GitHub\classification_models\evaluations"
root_data = r"C:\Users\esthe\Documents\GitHub\classification_models\data\training"
root_predict = r"C:\Users\esthe\Documents\GitHub\classification_models\data\training\prediction"
prediction_file = "Esther_MA\hierarchical-B_CNN-classification-use_model_structure-20241106_222713-z8l98z7u42_epoch9.pt-V1_0-train-20241109_220923-42.csv"
features_load_name = r'Esther_MA\feature_maps\flatten-vgg16-classification-flatten-20241105_210302_epoch0.pt-V1_0'


# %%
#Load feature vecotrs
with open(os.path.join(root_predict, features_load_name), "rb") as f_in:
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
all_labels = pd.read_csv(os.path.join(root_data, f'V1_0\metadata\streetSurfaceVis_v1_0.csv'), usecols=['mapillary_image_id', 'surface_type', 'surface_quality'])
all_labels = all_labels[~all_labels['surface_quality'].isna()]
all_labels = all_labels[~all_labels['surface_type'].isna()]



# %%
#adding true labels to our stored_df

stored_df = pd.merge(stored_df, all_labels, how="left", left_on="image_id",
                     right_on="mapillary_image_id")


# %%
#separating our stored_df in valid and training data
all_predictions = pd.read_csv(os.path.join(root_predict, prediction_file))
all_predictions = all_predictions.rename(columns = {"Image":"image_id"})
all_predictions['image_id'] = all_predictions['image_id'].astype('int64')
valid_predictions = all_predictions[all_predictions['is_in_validation'] == 1]
train_predictions = all_predictions[all_predictions['is_in_validation'] == 0]

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
train_df


# %%
validation_input_coarse_tsne = stored_coarse_features[valid_df['position'].to_list()]
validation_labels_coarse_tsne = valid_df['surface_type'].to_list()

train_input_coarse_tsne = stored_coarse_features[train_df['position'].to_list()]
train_labels_coarse_tsne = train_df['surface_type'].to_list()

validation_input_fine_tsne = stored_fine_features[valid_df['position'].to_list()]
validation_labels_fine_tsne = valid_df['surface_type'].to_list()

train_input_fine_tsne = stored_fine_features[train_df['position'].to_list()]
train_labels_fine_tsne = train_df['surface_type'].to_list()

validation_input_coarse_tsne


# %%


tsne_coarse_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15, random_state=seed).fit_transform(validation_input_coarse_tsne)
tsne_coarse_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_coarse_tsne)

tsne_fine_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15, random_state=seed).fit_transform(validation_input_fine_tsne)
tsne_fine_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_fine_tsne)


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
    plt.savefig(os.path.join(evaluation_path, f'{flag}_tsne_plot_validation.jpeg'))

# %%
create_plot(tsne_coarse_train, train_labels_coarse_tsne, 'train_coarse')
create_plot(tsne_coarse_valid, validation_labels_coarse_tsne, 'valid_coarse')

create_plot(tsne_fine_train, train_labels_fine_tsne, 'train_fine')
create_plot(tsne_fine_valid, validation_labels_fine_tsne, 'valid_fine')




# %%



