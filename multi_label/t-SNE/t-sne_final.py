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
import torch
from src import constants as const


# %%
#config = predict_config.B_CNN
level = const.FLATTEN
ds_type = "test"
seed=42
# evaluation_path = "/home/esther/surfaceai/classification_models/evaluations"
evaluation_path = r"\Users\esthe\Documents\GitHub\classification_models\evaluations\Esther_MA"
# root_data = "/home/esther/surfaceai/classification_models/data/training"
root_data = r"\Users\esthe\Documents\GitHub\classification_models\data\training"
# root_predict = os.path.join(root_data, "prediction", "Esther_MA")
root_predict = r"\Users\esthe\Documents\GitHub\classification_models\data\training\prediction\Esther_MA"
prediction_file = "hierarchical-B_CNN-classification-use_model_structure-20241106_222713-z8l98z7u42_epoch9.pt-classification-V1_0-test-20241123_134828-.csv"
features_load_name = "feature_maps/flatten-vgg16-classification-flatten-20241105_210302_epoch0.pt-V1_0"

# %%
#Load feature vecotrs
with open(os.path.join(root_predict, features_load_name), "rb") as f_in:
    #stored_data = torch.load(f_in)
    # print(type(stored_data))
    # print(stored_data)
    stored_data = pickle.load(f_in)
    #print(type(stored_data))
    #print(stored_data)
    stored_data.keys
    stored_ids = stored_data['image_ids']
    if level == const.HIERARCHICAL:
        stored_coarse_features = stored_data['coarse_features']
        stored_fine_features = stored_data['fine_features']
    else:
        stored_features = stored_data['features']
    stored_predictions = stored_data['pred_outputs']

# %%
if level == const.HIERARCHICAL:
    stored_df = pd.DataFrame({'image_id': stored_ids, 'coarse_features': str(stored_coarse_features),
                          'fine_features': str(stored_fine_features)})
else:
    stored_df = pd.DataFrame({'image_id': stored_ids,'features': str(stored_features)})
    

# %%
all_encodings = []  # Initialize an empty DataFrame
index = 0
for id in stored_ids:
    if level == const.HIERARCHICAL:
        coarse_feat = stored_coarse_features[index]
        fine_feat = stored_fine_features[index]
        data = {'image_id': int(id), 'fine_feat': str(fine_feat), 'coarse_feat': str(coarse_feat)}
    else:
        feat = stored_features[index]
        data = {'image_id': int(id), 'feat': str(feat)}
    #row_df = pd.DataFrame(data, index=[index])  # Create a DataFrame from the dictionary
    all_encodings.append(data)  # Append the row DataFrame to the main DataFrame
    index += 1
    
stored_df = pd.DataFrame(all_encodings)    

#print(stored_df)
    

# %%
#load the true labels
all_labels = pd.read_csv(os.path.join(root_data, f'V1_0/metadata/streetSurfaceVis_v1_0.csv'), usecols=['mapillary_image_id', 'surface_type', 'surface_quality'])
all_labels = all_labels[~all_labels['surface_quality'].isna()]
all_labels = all_labels[~all_labels['surface_type'].isna()]
all_labels['flatten_labels'] = all_labels.apply(lambda row: f"{row['surface_type']}__{row['surface_quality']}", axis=1)



#print(all_labels)
# %%
#adding true labels to our stored_df

stored_df = pd.merge(stored_df, all_labels, how="left", left_on="image_id",
                     right_on="mapillary_image_id")

# %%
#separating our stored_df in valid and training data
all_predictions = pd.read_csv(os.path.join(root_predict, prediction_file))
all_predictions = all_predictions.rename(columns = {"Image":"image_id"})
all_predictions['image_id'] = all_predictions['image_id'].astype('int64')
print(all_predictions)

valid_predictions = all_predictions[all_predictions['is_in_validation'] == 1]
train_predictions = all_predictions[all_predictions['is_in_validation'] == 0]

# %%
# merge all_predictions with stored_df
valid_df = pd.merge(stored_df, all_predictions[all_predictions['is_in_validation'] == 1],
                     how='inner', on='image_id')

train_test_df = pd.merge(stored_df, all_predictions[all_predictions['is_in_validation'] == 0],
                     how='inner', on='image_id')

print(valid_df)
print(train_test_df)

# %%
id_position = {image_id: position for position, image_id in enumerate(stored_ids)}
valid_df['image_id'] = valid_df['image_id'].astype('str')
valid_df['position'] = valid_df['image_id'].map(id_position)
train_test_df['image_id'] = train_test_df['image_id'].astype('str')
train_test_df['position'] = train_test_df['image_id'].map(id_position)
train_test_df


# %%
if level == const.HIERARCHICAL:
    validation_input_coarse_tsne = stored_coarse_features[valid_df['position'].to_list()]
    validation_labels_coarse_tsne = valid_df['surface_type'].to_list()

    train_input_coarse_tsne = stored_coarse_features[train_test_df['position'].to_list()]
    train_labels_coarse_tsne = train_test_df['surface_type'].to_list()

    validation_input_fine_tsne = stored_fine_features[valid_df['position'].to_list()]
    validation_labels_fine_tsne = valid_df['flatten_labels'].to_list()

    train_input_fine_tsne = stored_fine_features[train_test_df['position'].to_list()]
    train_labels_fine_tsne = train_test_df['flatten_labels'].to_list()
    
else:
    validation_input_tsne = stored_features[valid_df['position'].to_list()]
    validation_labels_tsne = valid_df['flatten_labels'].to_list()

    train_input_tsne = stored_features[train_test_df['position'].to_list()]
    train_labels_tsne = train_test_df['flatten_labels'].to_list()



# %%

if ds_type == "valid":
    if level == const.HIERARCHICAL:
        tsne_coarse_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(validation_input_coarse_tsne)
        tsne_fine_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(validation_input_fine_tsne)
    else:
        tsne_valid = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(validation_input_tsne)

elif ds_type == "train":
    if level == const.HIERARCHICAL:
        tsne_coarse_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_coarse_tsne)
        tsne_fine_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_fine_tsne)
    else:
        tsne_train = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_tsne)
    
else:
    if level == const.HIERARCHICAL:
        tsne_coarse_test = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_coarse_tsne)
        tsne_fine_test = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_fine_tsne)
    else:
        tsne_test = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=seed).fit_transform(train_input_tsne)

# %%
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def generate_color_palette(num_colors):
    # Generate a set of distinguishable colors
    colors = sns.color_palette("hsv", num_colors)
    return colors

def create_and_save_plot(tsne_data, tsne_label, save_name, labels_subset=None):
    
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
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(scatter_labels_encoded)):
        indices = np.where(scatter_labels_encoded == label)
        plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1], c=[colors[i]], label=label_encoder.classes_[label], s=10)

    plt.title(f't-SNE {save_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(os.path.join(evaluation_path, f'{save_name}_tsne_plot_{prediction_file}.jpeg'))
    print(f'{save_name} plot saved')
    plt.show()

# %%
if level == const.HIERARCHICAL:
    if ds_type == "valid":
        create_and_save_plot(tsne_coarse_valid, validation_labels_coarse_tsne, 'valid_coarse', labels_subset=['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved'])
        create_and_save_plot(tsne_fine_valid, validation_labels_fine_tsne, 'valid_fine', labels_subset=['asphalt__excellent','asphalt__good','asphalt__intermediate','asphalt__bad',
                                                                                        'concrete__excellent','concrete__good','concrete__intermediate','concrete__bad',
                                                                                        'paving_stones__excellent','paving_stones__good','paving_stones__intermediate','paving_stones__bad',
                                                                                        'sett__good','sett__intermediate','sett__bad',
                                                                                        'unpaved__intermediate','unpaved__bad','unpaved__very_bad',
                                                                                        ])
    elif ds_type == "train":
        create_and_save_plot(tsne_coarse_train, train_labels_coarse_tsne, 'train_coarse')
        create_and_save_plot(tsne_fine_train, train_labels_fine_tsne, 'train_fine')


    elif ds_type == "test":
        create_and_save_plot(tsne_coarse_test, train_labels_coarse_tsne, 'test_coarse', labels_subset=['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved'])
        create_and_save_plot(tsne_fine_test, train_labels_fine_tsne, 'test_fine')
        
else:
    if ds_type == "valid":
        create_and_save_plot(tsne_valid, validation_labels_tsne, 'valid')
    elif ds_type == "train":
        create_and_save_plot(tsne_train, train_labels_tsne, 'train')
    else:
        create_and_save_plot(tsne_test, train_labels_tsne, 'test')
    # create_and_save_plot(tsne_coarse_test, train_labels_coarse_tsne, 'test_coarse', labels_subset=['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved'])
    # create_and_save_plot(tsne_fine_test, train_labels_fine_tsne, 'test_fine', labels_subset=['asphalt__excellent','asphalt__good','asphalt__intermediate','asphalt__bad',
    #                                                                                    'concrete__excellent','concrete__good','concrete__intermediate','concrete__bad',
    #                                                                                    'paving_stones__excellent','paving_stones__good','paving_stones__intermediate','paving_stones__bad',
    #                                                                                    'sett__good','sett__intermediate','sett__bad',
    #                                                                                    'unpaved__intermediate','unpaved__bad','unpaved__very_bad',
    #                                                                                    ])
# create_and_save_plot(tsne_coarse_train, train_labels_coarse_tsne, 'train_coarse')

# create_and_save_plot(tsne_fine_train, train_labels_fine_tsne, 'train_fine')



# def create_plot(tsne_data, tsne_label, flag):
#     label_encoder = LabelEncoder()
#     scatter_labels_encoded = label_encoder.fit_transform(tsne_label)

#     # Create a scatter plot
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=scatter_labels_encoded, cmap='viridis', s=10)
#     plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Surface Type').set_ticklabels(label_encoder.classes_)
#     plt.title('t-SNE Feature Visualization')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.savefig(os.path.join(evaluation_path, f'{flag}_tsne_plot_{prediction_file}.jpeg'))
#     print(f'{flag} plot saved')
#     plt.show()

# # %%
# if ds_type == "valid":
#     create_plot(tsne_coarse_valid, validation_labels_coarse_tsne, 'valid_coarse')
#     create_plot(tsne_fine_valid, validation_labels_fine_tsne, 'valid_fine')

# elif ds_type == "train":
#     create_plot(tsne_coarse_train, train_labels_coarse_tsne, 'train_coarse')
#     create_plot(tsne_fine_train, train_labels_fine_tsne, 'train_fine')
    
# else:
#     create_plot(tsne_coarse_test, train_labels_coarse_tsne, 'test_coarse')
#     create_plot(tsne_fine_test, train_labels_fine_tsne, 'test_fine')




# %%



