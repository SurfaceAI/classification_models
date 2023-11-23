from ast import literal_eval
import cv2 as cv
import numpy as np
import tensorflow as tf
import argparse
import sys
import os.path
import random
import os
import glob
import operator
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#
# 1. you have to move your model files to the other folder
# 2. Create checkpoint files?
# 3. Execute pretrained models

execution_directory = os.path.dirname(os.path.abspath(__file__))
import dataset

image_size=128
num_channels=3
images = []



#restoring the models

graph = tf.Graph()
graphAQ = tf.Graph()
graphPQ = tf.Graph()

default_graph = tf.compat.v1.get_default_graph()
type_folder = os.path.join(os.path.dirname(execution_directory), '01Surface Type')


# ----------------------------- #
# Restoring the model for types #
# ----------------------------- #


with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(f'{execution_directory}/roadsurface-model.meta')
    # Acessing the graph
    #
    y_pred = graph.get_tensor_by_name("y_pred:0")

    #
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir(f'{type_folder}/train_data'))))

sess = tf.compat.v1.Session(graph = graph)
tf.compat.v1.disable_eager_execution()
saver.restore(sess, tf.train.latest_checkpoint(f'{execution_directory}/typeCheckpoint/'))



# --------------------------------------- #
# Restoring the model for asphalt quality #
# --------------------------------------- #

with graphAQ.as_default():
    saverAQ = tf.compat.v1.train.import_meta_graph(f'{execution_directory}/roadsurfaceAsphaltQuality-model.meta')
    # Acessing the graph
    #
    y_predAQ = graphAQ.get_tensor_by_name("y_pred:0")

    #
    xAQ = graphAQ.get_tensor_by_name("x:0")
    y_trueAQ = graphAQ.get_tensor_by_name("y_true:0")
    y_test_imagesAQ = np.zeros((1, len(os.listdir(f'{execution_directory}/training_data_asphalt_quality'))))

sessAQ = tf.compat.v1.Session(graph = graphAQ)
saverAQ.restore(sessAQ, tf.train.latest_checkpoint(f'{execution_directory}/asphaltCheckpoint/'))



# ------------------------------------- #
# Restoring the model for paved quality #
# ------------------------------------- #
with graphPQ.as_default():
    saverPQ = tf.compat.v1.train.import_meta_graph(f'{execution_directory}/roadsurfacePavedQuality-model.meta')
    # Acessing the graph
    #
    y_predPQ = graphPQ.get_tensor_by_name("y_pred:0")

    #
    xPQ = graphPQ.get_tensor_by_name("x:0")
    y_truePQ = graphPQ.get_tensor_by_name("y_true:0")
    y_test_imagesPQ = np.zeros((1, len(os.listdir(f'{execution_directory}/training_data_paved_quality'))))

sessPQ = tf.compat.v1.Session(graph = graphPQ)
saverPQ.restore(sessPQ, tf.train.latest_checkpoint(f'{execution_directory}/pavedCheckpoint/'))


# needed when we add a third surface type class, e.g. unpaved

# # --------------------------------------- #
# # Restoring the model for unpaved quality #
# # --------------------------------------- #
# with graphUQ.as_default():
#     saverUQ = tf.train.import_meta_graph('roadsurfaceUnpavedQuality-model.meta')
#     # Acessing the graph
#     #
#     y_predUQ = graphUQ.get_tensor_by_name("y_pred:0")

#     #
#     xUQ = graphUQ.get_tensor_by_name("x:0")
#     y_trueUQ = graphUQ.get_tensor_by_name("y_true:0")
#     y_test_imagesUQ = np.zeros((1, len(os.listdir('training_data_unpaved_quality'))))

# sessUQ = tf.Session(graph = graphUQ)
# saverUQ.restore(sessUQ, tf.train.latest_checkpoint('unpavedCheckpoint/'))


#loading test images, again as DataSet object:

#asphalt 

paved_images_dir = f'{execution_directory}/test_data_asphalt_quality'
classesAQ = os.listdir(paved_images_dir)
num_classes = len(classesAQ)
img_size= 128

test_data_asphalt = dataset.read_test_sets(paved_images_dir, img_size, classesAQ)

test_images_asphalt = test_data_asphalt.images


#paved

paved_images_dir = f'{execution_directory}/test_data_paved_quality'
classesPQ = os.listdir(paved_images_dir)
num_classes = len(classesAQ)
img_size= 128


test_data_paved = dataset.read_test_sets(paved_images_dir, img_size, classesPQ)
test_images_paved = test_data_paved.images

#initializing the lists for our labels
pred_class_type = []
true_class_type = []
pred_classQ = []
true_classQ = []

test_data_sets = [test_data_asphalt, test_data_paved]


for test_data in test_data_sets:
    
    for i, image in enumerate(test_data.images): 
        
        true_index = None
        
        x_batch = image.reshape(1, image_size, image_size, num_channels)
        
        #feed images in the model and get predicted type label
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        
        outputs = [result[0,0], result[0,1]]

        
        value = max(outputs)
        index = np.argmax(outputs)
        pred_class_type.append(index)
    
        #accessing true type label
        if test_data == test_data_asphalt:
             true_index = 0
             true_indexAQ = np.argmax(test_data.labels[i])
             true_classQ.append(true_indexAQ)
        else:
             true_index = 1
             true_indexPQ = np.argmax(test_data.labels[i])
             true_classQ.append(true_indexPQ)
        true_class_type.append(true_index)


        if index == 0: #Asphalt
            label = 'paved(asphalt)'
            prob = str("{0:.2f}".format(value))
            color = (0, 0, 0)
            x_batchAQ = image.reshape(1, image_size, image_size, num_channels)
            
            #prediction of asphalt quality label
            feed_dict_testingAQ = {xAQ: x_batchAQ, y_trueAQ: y_test_imagesAQ}
            resultAQ = sessAQ.run(y_predAQ, feed_dict=feed_dict_testingAQ)
            outputsQ = [resultAQ[0,0], resultAQ[0,1], resultAQ[0,2]]
            valueQ = max(outputsQ)
            indexQ = np.argmax(outputsQ)
            pred_classQ.append(indexQ)
            
            if indexQ == 0: #Asphalt - Excellent
                quality = 'Excellent'
                colorQ = (0, 255, 0)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 1: #Asphalt - Good
                quality = 'Good'
                colorQ = (0, 204, 255)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 2: #Asphalt - Intermediate
                quality = 'Intermediate'
                colorQ = (0, 0, 255)
                probQ =  str("{0:.2f}".format(valueQ))  
                
        elif index == 1: #Paved
            label = 'paved(concrete)'
            prob = str("{0:.2f}".format(value))
            color = (153, 102, 102)
            x_batchPQ = image.reshape(1, image_size, image_size, num_channels)
            #
            feed_dict_testingPQ = {xPQ: x_batchPQ, y_truePQ: y_test_imagesPQ}
            resultPQ = sessPQ.run(y_predPQ, feed_dict=feed_dict_testingPQ)
            outputsQ = [resultPQ[0,0], resultPQ[0,1], resultPQ[0,2]]
            valueQ = max(outputsQ)
            indexQ = np.argmax(outputsQ)
            pred_classQ.append(indexQ)
            
            if indexQ == 0: #Paved - Bad
                quality = 'Bad'
                colorQ = (0, 255, 0)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 1: #Paved - Intermediate
                quality = 'Intermediate'
                colorQ = (0, 204, 255)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 2: #Paved - Very bad
                quality = 'Very bad'
                colorQ = (0, 0, 255)
                probQ =  str("{0:.2f}".format(valueQ))
                  
        # print(f"Image: {image}, Class: {label}, Probability: {prob}")
        print(f"Class: {label}, Probability: {prob}, Quality: {quality}")
     
# cv.rectangle(finalimg, (0, 0), (145, 80), (255, 255, 255), cv.FILLED)
# cv.putText(finalimg, 'Class: ', (5,15), cv.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
# cv.putText(finalimg, label, (70,15), cv.FONT_HERSHEY_DUPLEX, 0.5, color)
# cv.putText(finalimg, prob, (5,35), cv.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
# cv.putText(finalimg, 'Quality: ', (5,55), cv.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
# cv.putText(finalimg, quality, (70,55), cv.FONT_HERSHEY_DUPLEX, 0.5, colorQ)
# cv.putText(finalimg, probQ, (5,75), cv.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))


sess.close()
sessAQ.close()
sessPQ.close()


# Hier muss ich noch die Performance evaluation für beide Labels kombiniert hinzufügen. 

true_labels_combined = np.column_stack((true_class_type, true_classQ))
pred_labels_combined = np.column_stack((pred_class_type, pred_classQ))

# somehow this is not working
# from sklearn.metrics import multilabel_confusion_matrix
# multilabel_confusion_matrix(true_labels_combined, pred_labels_combined)

correct_pred = true_labels_combined == pred_labels_combined
correct_pred_count = np.sum(np.all(correct_pred, axis=1))
print(correct_pred_count)



#not for multiclass
# accuracy = accuracy_score(true_labels_combined, pred_labels_combined)
# print(f"Accuracy: {accuracy}")


# conf_matrix = confusion_matrix(true_class, pred_class)

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)

# # Calculate accuracy
# accuracy = accuracy_score(true_class, pred_class)
# print(f"Accuracy: {accuracy}")

# # Calculate precision, recall, and F1-score
# report = classification_report(true_class, pred_class)
# print("Classification Report:")
# print(report)

# # Visualize the confusion matrix using a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=["paved(asphalt)", "paved(concrete)"],
#             yticklabels=["paved(asphalt)", "paved(concrete)"])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Class")
# plt.ylabel("True Class")
# plt.show()