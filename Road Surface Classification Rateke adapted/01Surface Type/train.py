import cv2
import tensorflow as tf
from datetime import timedelta
import numpy as np
import os
import mlflow
import shutil
#from mlflow.models import infer_signature


# This is the directory in which this .py file is in
execution_directory = os.path.dirname(os.path.abspath(__file__))


#mlflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

#First set which model and which data preprocessing to include, either just cropped or cropped and augmented
#Models: roadsurface-model.meta; roadsurface-model-augmented.meta
#Dataset files: "dataset", "dataset_augmented"

train_path = os.path.join(execution_directory, 'train_data')
save_path = execution_directory
quality_path = os.path.join(os.path.dirname(execution_directory), '02Surface Quality') #our surface quality folder
model = "roadsurface-model"
dataset = "dataset"
import dataset



#defining hypterparameters and input data
batch_size = 32
validation_size = 0.2
learning_rate = 1e-4
img_size = 128
num_channels = 3


#Adding global tensorflow seed 
tf.random.set_seed(0)

#os.system('spd-say -t male3 "I will try to learn this, my master."')

#Prepare input data
classes = os.listdir(train_path) 
num_classes = len(classes)
classes

# We shall load all the train and validation images and labels into memory using openCV and use that during train
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

#Here, we print the first image of our train data after preprocessing to check how it looks.
# It should pop up in an image editor outside of this window. 
# cv2.imshow('image view',data.valid.images[0])
# k = cv2.waitKey(0) & 0xFF #without this, the execution would crush the kernel on windows
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
#first images of train and valid datasets are always the same

#Alternatively, we can also save our images in separate folders in our directory
#save_folder = 'preprocessed_images'


# # Assuming data is an instance of DataSet
# for i in range(data.train.num_examples):
#     image = data.train.images[i].squeeze()  # Remove the batch dimension
#     dataset.save_image(image, save_folder, f"image_{i}")
    

    
# Initialize session
session = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution() 

x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.compat.v1.argmax(y_true, axis=1)


##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape, seed=None):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.05, seed=seed))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters], seed=1)
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filters=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling. Why padding='SAME', when we want to reduce the size?
    layer = tf.nn.max_pool(layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], #[batch_stride, x-stride, y-stride, depth-stride]
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer



def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs], seed=1)
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 



#Prediction
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.compat.v1.argmax(y_pred, dimension=1)


#Optimization

session.run(tf.compat.v1.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.compat.v1.global_variables_initializer()) 



def show_progress(epoch, feed_dict_train, train_loss, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Training Loss: {2:.3f}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1, acc, train_loss, val_acc, val_loss))

total_iterations = 0

saver = tf.compat.v1.train.Saver()

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):
        
        tf.random.set_seed(0)

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            train_loss = session.run(cost, feed_dict=feed_dict_tr)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            # mlflow.log_metric("train_loss", train_loss, step=i)
            # mlflow.log_metric("val_loss", val_loss, step=i)
            
            # mlflow.log_param("epoch", epoch)
            # mlflow.log_param("learning_rate", learning_rate)
            # mlflow.log_param("batch_size", batch_size)
            
            # mlflow.tensorflow.log_model(saver, "mlruns")
                        
            show_progress(epoch, feed_dict_tr, train_loss, feed_dict_val, val_loss)
            saver.save(session, f'{save_path}/{model}') 
            saver.save(session, f'{quality_path}/{model}') #saving model also in out Surface Quality folder as we need it for the combined prediction later



    total_iterations += num_iteration


train(num_iteration=100)



# with mlflow.start_run(run_name='new try'):
#     # Train the model
#     train(num_iteration=100)
    
#     # Disable eager execution for TensorFlow
#     tf.compat.v1.disable_eager_execution()
    
#     # Import the TensorFlow graph
#     saver = tf.compat.v1.train.import_meta_graph(f'{execution_directory}/{model}.meta')
#     model_restore = saver.restore(session, tf.compat.v1.train.latest_checkpoint(f'{execution_directory}/'))
    
#     # Log metrics to MLflow
#     #mlflow.log_metric("train_loss", train_loss)
#     #mlflow.log_metric("val_loss", val_loss)
#     #mlflow.log_param("num_iteration", 100)
    
#     # Log TensorFlow model
#     mlflow.tensorflow.log_model(saver, "mlruns")



#saving our model files to the folder '02Surface Quality' so we can access it more easily for the combined prediction
model_files = [f'{model}.meta', f'{model}.index', f'{model}.data-00000-of-00001']

for file in model_files:
    source_path = os.path.join(execution_directory, file)
    target_path = os.path.join(quality_path, file)
    shutil.copy2(source_path, target_path)

#os.system('spd-say -t male3 "I have finished my train. Lets do it!"')
print('\a')



