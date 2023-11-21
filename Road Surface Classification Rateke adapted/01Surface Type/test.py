#!/usr/bin/env python


from dataset import read_test_sets
import numpy as np
import tensorflow as tf
import os.path
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#current directory
execution_directory = os.path.dirname(os.path.abspath(__file__))

test_path = os.path.join(execution_directory, 'test_data')
save_path = execution_directory

save_path

tf.compat.v1.reset_default_graph()

model = 'roadsurface-model-pretrained'
image_size=128
num_channels=3
images = []


# Restoring the model
sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()
saver = tf.compat.v1.train.import_meta_graph(f'{save_path}/{model}.meta')
saver.restore(sess, tf.train.latest_checkpoint(f'{save_path}/'))

# Acessing the graph
graph = tf.compat.v1.get_default_graph()

# Getting tensors from the graph
y_pred = graph.get_tensor_by_name("y_pred:0")

#
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(os.listdir(test_path))))



#loading test images, again as DataSet object:

test_images_dir = test_path
classes = os.listdir(test_path)
num_classes = len(classes)
img_size= 128

test_data = read_test_sets(test_images_dir, img_size, classes)

test_images = test_data.images
test_images.shape


#preprocess images exactly as done before

pred_class = []
true_class = []

for i, image in enumerate(test_data.images): 
    
    
    x_batch = image.reshape(1, image_size, image_size, num_channels)
    
    #feed images in the model
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    
    outputs = [result[0,0], result[0,1]]

    value = max(outputs)
    index = np.argmax(outputs)
    pred_class.append(index)
    
    #accessing true label
    true_index = np.argmax(test_data.labels[i])
    true_class.append(true_index)

    if index == 0:
        label = 'paved(asphalt)'
        prob = str("{0:.2f}".format(value))
        color = (0, 0, 0)
    elif index == 1:
        label = 'paved(concrete)'
        prob = str("{0:.2f}".format(value))
        color = (153, 102, 102)
    # elif index == 2:
    #     label = 'Unpaved'
    #     prob = str("{0:.2f}".format(value))
    #     color = (0, 153, 255)
        
    # print(f"Image: {image}, Class: {label}, Probability: {prob}")
    print(f"Class: {label}, Probability: {prob}")
     

sess.close()


#print evaluation parameters

accuracy = accuracy_score(true_class, pred_class)
print(f"Accuracy: {accuracy}")


conf_matrix = confusion_matrix(true_class, pred_class)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(true_class, pred_class)
print(f"Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score
report = classification_report(true_class, pred_class)
print("Classification Report:")
print(report)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["paved(asphalt)", "paved(concrete)"],
            yticklabels=["paved(asphalt)", "paved(concrete)"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
