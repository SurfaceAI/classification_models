import os
import random
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, top_k_accuracy_score
from metrics import minimum_sensitivity, accuracy_off1
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.applications.vgg16 import VGG16
from clm import CLM
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import History, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from losses import categorical_ce_exponential_regularized, categorical_ce_poisson_regularized, categorical_ce_binomial_regularized, qwk_loss, make_cost_matrix, categorical_ce_beta_regularized, ordinal_distance_loss_base, ordinal_distance_loss, ordinal_distance_loss_hier, qwk_loss_hier, qwk_loss_base
from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.utils import Sequence
import albumentations as A
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.utils import shuffle
from wandb.keras import WandbCallback
import datetime

def preprocessing(path_imgs, csv, parte, colormode, img_rows, img_cols, use_metaclasses, cnn)-> 'tuple[np.ndarray, np.ndarray]':

    if cnn == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif cnn == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet import preprocess_input

    # original_class : [global_class, major_class, minor_class]
    labels_mapping = {'1': [0, 0, 0], '2-': [1, 1, 0], '2': [2, 1, 1], '2+': [3, 1, 2], '3-': [
        4, 2, 0], '3': [5, 2, 1], '3+': [6, 2, 2], '4-': [7, 3, 0], '4': [8, 3, 1], '4+': [9, 3, 2]}

    imgs_array = []
    label_array = []

    columns = [x for x in csv.columns.values if parte in x]
    imgs_sx = csv[csv[columns[0]].isin(labels_mapping.keys())]
    imgs_dx = csv[csv[columns[1]].isin(labels_mapping.keys())]

    if use_metaclasses == True:
        imgs_sx[columns[0]] = imgs_sx[columns[0]].str.slice_replace(1, repl='')
        imgs_dx[columns[1]] = imgs_dx[columns[1]].str.slice_replace(1, repl='')

    for index, row in imgs_sx.iterrows():
        name = row['IMG_LATOSX']
        try:
            # Load image from path
            image = load_img(os.path.join(path_imgs, name),
                             target_size=(img_rows, img_cols), color_mode=colormode)
        except Exception:
            print('{}_not found'.format(name))
            continue

        # Load labels
        label = row[columns[0]]
        # Convert named label (1, 2-, 2+...) to [global, major, minor] (0-9, 0-4, 0-3)
        full_label = labels_mapping[label]
        # full_label, group
        label_array.append(full_label + [int(row['ID'])])

        # Preprocess image and add it to the imgs array
        x = img_to_array(image)
        if colormode == "grayscale":
            x = np.array(np.dstack((x, x, x)), dtype=np.uint8)
        x = preprocess_input(x)
        imgs_array.append(x)

    for index, row in imgs_dx.iterrows():
        name = row['IMG_LATODX']
        try:
            image = load_img(os.path.join(path_imgs, name),
                             target_size=(img_rows, img_cols), color_mode=colormode)
        except Exception:
            print('{}_not found'.format(name))
            continue

        # Load labels
        label = row[columns[0]]
        # Convert named label (1, 2-, 2+...) to [named, global, major, minor] (0-9, 0-4, 0-3)
        full_label = labels_mapping[label]
        # full_label, group
        label_array.append(full_label + [int(row['ID'])])

        # Preprocess image and add it to the imgs array
        x = img_to_array(image)
        if colormode == "grayscale":
            x = np.array(np.dstack((x, x, x)), dtype=np.uint8)
        x = preprocess_input(x)
        imgs_array.append(x)

    X = np.array(imgs_array)
    y = np.array(label_array)

    return X, y


def fix_seeds(seed: int) -> None:
    """ Fix random seeds for numpy, tensorflow, random, etc.

        Parameters
        -----------
        seed : int.
            Random seed.
    """

    np.random.seed(seed)  # numpy seed
    tf.random.set_seed(seed)  # tensorflow seed
    random.seed(seed)  # random seed
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)

    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass


def compute_metrics(y_true, y_pred, num_classes):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    labels = range(0, num_classes)

    # Calculate metrics
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
    ms = minimum_sensitivity(y_true, y_pred, labels=labels)
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    off1 = accuracy_off1(y_true, y_pred, labels=labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        'QWK': qwk,
        'MS': ms,
        'MAE': mae,
        'CCR': acc,
        '1-off': off1,
        'Confusion matrix': conf_mat
    }

    return metrics


def print_metrics(metrics):
		print('Confusion matrix :\n{}'.format(metrics['Confusion matrix']))
		print('QWK: {:.4f}'.format(metrics['QWK']))
		print('MS: {:.4f}'.format(metrics['MS']))
		print('MAE: {:.4f}'.format(metrics['MAE']))
		print('CCR: {:.4f}'.format(metrics['CCR']))
		print('1-off: {:.4f}'.format(metrics['1-off']))
  

class PrintThresholdsCallback(Callback):
    def __init__(self, model, layer_name):
        super(PrintThresholdsCallback, self).__init__()
        self.model = model
        self.layer_name = layer_name

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        if hasattr(layer, 'thresholds_b') and hasattr(layer, 'thresholds_a'):
            thresholds_b = layer.thresholds_b.numpy()
            thresholds_a = layer.thresholds_a.numpy()
            thresholds = layer._convert_thresholds(thresholds_b, thresholds_a, layer.min_distance)
            print(f'Epoch {epoch + 1} - Thresholds: {thresholds}')
        else:
            print(f'Epoch {epoch + 1} - Fixed Thresholds: {layer.thresholds}')
            
class GradientLoggingCallback(Callback):
    def __init__(self, log_dir):
        super(GradientLoggingCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel
                    grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
                    for grad, weight in zip(grads, weights):
                        tf.summary.histogram(weight.name, weight, step=epoch)
                        tf.summary.histogram(weight.name + '/grad', grad, step=epoch)
        self.writer.flush()


def create_vgg16_model(img_shape: np.ndarray, n_labels: int, trainable_convs: bool, clm: 'dict[str, Union[float, bool]]', obd:dict)-> Model:
    # Get the VGG16 pretrained model and set layers trainable status
    vgg16_conv = VGG16(include_top=False, weights='imagenet',
                       input_shape=(img_shape[0], img_shape[1], 3))
    if not trainable_convs:
        for layer in vgg16_conv.layers[:-1]:
            layer.trainable = True

    if obd['enabled']:
        # build top model         
        x = Flatten(name='flatten')(vgg16_conv.output)

        hidden_size_per_unit = np.round(4096 / (n_labels - 1)).astype(int)

        layers = []
        for i in range(n_labels - 1):
            x1 = Dense(hidden_size_per_unit, name='hidden_{}'.format(i))(x)
            x1 = LeakyReLU()(x1)
            x1 = Dropout(0.3)(x1)
            x1 = Dense(1, name='out_{}'.format(i))(x1)
            x1 = BatchNormalization()(x1) 
            x1 = Activation('sigmoid')(x1)
            layers.append(x1)

        # stitch together
        out = Concatenate(axis=1)(layers) #([tf.expand_dims(o,axis=1) for o in layers])
        model = Model(inputs= vgg16_conv.input, outputs=out)

    else:
        # Create top layers
        x = Flatten(name='flatten')(vgg16_conv.output)
        x = Dropout(0.5)(x)
        x = Dense(4096, name='fc1')(x)
        x = Activation('relu')(x)
        x = Dense(4096, name='fc2')(x)
        x = Activation('relu')(x)

        if clm['enabled']:
            x = Dense(1, dtype='float32')(x)
            x = BatchNormalization()(x)
            x = CLM(num_classes=n_labels, link_function=clm['link'], min_distance=clm['min_distance'],
                    use_slope=clm['use_slope'], fixed_thresholds=clm['fixed_thresholds'])(x)
        else:
            x = Dense(n_labels, dtype='float32')(x)
            x = BatchNormalization()(x)
            x = Activation('softmax')(x)

        # Create full model
        model = Model(inputs=vgg16_conv.input, outputs=x)

    return model


def compute_splits_hash(splits: 'list[tuple[np.ndarray, np.ndarray]]')-> str:
    """
    Computes a hash based on the indices of the splits.
    This hash can be used to identify this exact split
    """

    import hashlib

    hashstr = ""
    for tr_idx, t_idx in splits:
        tr_str = ','.join([str(index) for index in tr_idx])
        t_str = ','.join([str(index) for index in t_idx])
        hashstr += tr_str + t_str

    return hashlib.md5(hashstr.encode('utf-8')).hexdigest()


def run_cnn(train_data: 'tuple[np.ndarray, np.ndarray]', validation_data: 'tuple[np.ndarray, np.ndarray]',
            test_data: 'tuple[np.ndarray, np.ndarray]', optimiser_params: dict, clm: dict, obd: dict, loss_config: dict,
            trainable_convs: bool = False, labels = None, return_labels = True, augment: bool = True):
    # Get data X and y
    # y contains only an integer representing the class number
    X_train, y_train = train_data
    X_val, y_val = validation_data
    X_test, y_test = test_data

    # Get image shape from data
    img_shape = X_train.shape[1:3]

    # If labels param is not set, get labels from data
    if labels is None:
        labels = np.unique(y_train)

    n_labels = len(labels)
        

    # Compute class weights from class counts
    # class_weights = class_weight.compute_class_weight(
    #     'balanced', classes=labels, y=y_train)

    if obd['enabled']:
        target_class_np, target_class_tf = binary_conv(n_labels)

        y_train = tf.gather(target_class_tf, y_train)
        y_val = tf.gather(target_class_tf, y_val)
        y_test = tf.gather(target_class_tf, y_test)
        
        # Define the loss 
        if loss_config['type'] == 'mae':
            loss = ordinal_distance_loss_base('mae',n_labels)
        elif loss_config['type'] == 'mse':
            loss = ordinal_distance_loss_base('mse',n_labels)
        else:
            raise ValueError('Accepted losses are mae and mse')
        
        metric = 'mae'
    else:
        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)

        # Define the loss 
        if loss_config['type'] == 'exp':
            loss = categorical_ce_exponential_regularized(
                n_labels, eta=loss_config['eta'], tau=loss_config['tau'], l=loss_config['l'])
        elif loss_config['type'] == 'poi':
            loss = categorical_ce_poisson_regularized(
                n_labels, eta=loss_config['eta'])
        elif loss_config['type'] == 'bin':
            loss = categorical_ce_binomial_regularized(
                n_labels, eta=loss_config['eta'])
        elif loss_config['type'] == 'beta':
            loss = categorical_ce_beta_regularized(n_labels, eta=loss_config['eta'])
        elif loss_config['type'] == 'qwk':
            cost_matrix = make_cost_matrix(n_labels)
            loss = qwk_loss_base(cost_matrix)
        else:
            loss = 'categorical_crossentropy'

        metric_1 = CohenKappa(n_labels, weightage='quadratic', name='qwk')
        metric_2 = 'acc'

    # Create data augmentation generator
    aug = ImageDataGenerator(horizontal_flip=augment,
                             fill_mode="constant", cval=0.0)
    generator = aug.flow(X_train, y_train,
                         batch_size=optimiser_params['bs'])

    # Create vgg16 full model
    model = create_vgg16_model(img_shape, n_labels, trainable_convs, clm, obd)

    # Define the optimiser and compile the model
    optimiser = Adam(lr=optimiser_params['lr'])
    model.compile(loss=loss, optimizer=optimiser, metrics=[metric_1, metric_2])

    # Define the callbacks
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #gradient_logging_callback = GradientLoggingCallback(log_dir)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    print_thresholds = PrintThresholdsCallback(model, layer_name="clm")
    wandb_callback = WandbCallback(log_gradients=True, training_data=(X_train, y_train), save_model=False)
    #wandb_callback = WandbCallback(log_weights=True, log_gradients=True, training_data=(X_train, y_train), save_model=False)
    callbacks = [early_stopping, print_thresholds, wandb_callback]


    model.summary()

    # Train the model
    history: History = model.fit(
        generator,
        epochs=optimiser_params['epochs'],
        steps_per_epoch=generator.__len__(),
        validation_data=(X_val, y_val),
        verbose=1,
        #class_weight={c: w for c, w in enumerate(class_weights)},
        callbacks=callbacks
    )

    # Predict on train, validation and test
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    if obd['enabled']:
        distances = cdist(y_test_pred, target_class_np, metric='euclidean')
        if return_labels:
            # Return predictions
            print(distances.argmin(axis=1))
            return distances.argmin(axis=1)
        else:
            # return probabilities
            return softmax(-distances, axis=1)
    else:    
        if return_labels:
            # Return predictions
            return np.argmax(y_test_pred, axis=1)
        else:
            # return probabilities
            return y_test_pred

def binary_conv(num_classes):
    target_class = np.ones((num_classes, num_classes-1), dtype=np.float32)
    target_class[np.triu_indices(num_classes, 0, num_classes-1)] = 0.0
    target_class_np = target_class
    target_class_tf = tf.convert_to_tensor(target_class, dtype=tf.float32)

    return target_class_np, target_class_tf


def aug():
	train_transform = [

		A.HorizontalFlip(p=0.5)
	]
	return A.Compose(train_transform)


class batch_generator(Sequence):
    def __init__(self, x, y, batch_size, img_size, augmentation=True, target_class_macro=None, target_class_micro=None, seed=1):
    
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.img_size = img_size
        self.augmentation = augmentation
        self.target_class_macro = target_class_macro
        self.target_class_micro = target_class_micro
        self.classes_macro = len(np.unique(y[:,1]))
        self.classes_micro = len(np.unique(y[:,2]))
        self.seed = seed


    def __len__(self):
        return int(np.ceil(len(self.x) // self.batch_size))

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.x[i : i + self.batch_size]
        batch_target_1 = self.y[:,1][i : i + self.batch_size]
        batch_target_2 = self.y[:,2][i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y1 = np.zeros((self.batch_size,), dtype = np.int32)
        y2 = np.zeros((self.batch_size,), dtype = np.int32)

        for (j, img, label1, label2) in zip(range(len(batch_input_img_paths)),batch_input_img_paths, batch_target_1, batch_target_2):
            # apply augmentations
            if self.augmentation:
                img = aug()(image=img)['image']

            x[j] = img
            y1[j] = label1
            y2[j] = label2
        #print(np.unique(y1))
        y_tot = np.column_stack((y1, y2))

        #if self.target_class_macro is not None:
        #    y_macro = tf.gather(self.target_class_macro, y1)
        #    y_micro = tf.gather(self.target_class_micro, y2)
        #else:
        #    y_macro = to_categorical(y1,num_classes=self.classes_macro)
        #    y_micro = to_categorical(y2,num_classes=self.classes_micro)    
        #return x, {'macro_output': y_macro, 'micro_output': [y1,y_micro]}

        return x, {'macro_output': y1, 'micro_output': y_tot}


    def on_epoch_end(self):
        self.x , self.y = shuffle(self.x, self.y, random_state=self.seed)


def build_standard_branch(x, inputs, n_labels, head, name):

    x = Flatten(name='flatten_{}'.format(name))(x)

    if head['name'] == 'obd':
        hidden_size_per_unit = np.round(inputs / (n_labels - 1)).astype(int)
        layers = []
        for i in range(n_labels - 1):
            x1 = Dense(hidden_size_per_unit, name='hidden_{}_{}'.format(name,i))(x)
            x1 = LeakyReLU()(x1)
            x1 = Dropout(0.3)(x1)
            x1 = Dense(1, name='out_{}_{}'.format(name,i))(x1)
            x1 = BatchNormalization()(x1) 
            x1 = Activation('sigmoid')(x1)
            layers.append(x1)

        # stitch together
        out = Concatenate(axis=1,name="{}_output".format(name))(layers)

    else:
        # Create top layers
        x = Dropout(0.5)(x)
        x = Dense(4096, name='fc1_{}'.format(name))(x)
        x = Activation('relu')(x)
        x = Dense(4096, name='fc2_{}'.format(name))(x)
        x = Activation('relu')(x)

        if head['name'] == 'clm':
            clm = head

            x = Dense(1, dtype='float32')(x)
            x = BatchNormalization()(x)
            out_name = "{}_output".format(name)
            out = CLM(num_classes=n_labels, link_function=clm['link'], name = out_name, min_distance=clm['min_distance'],
                    use_slope=clm['use_slope'], fixed_thresholds=clm['fixed_thresholds'])(x)
        else:
            x = Dense(n_labels, dtype='float32')(x)
            x = BatchNormalization()(x)
            out = Activation('softmax')(x)
            out._name = "{}_output".format(name)

    return out


def build_deep_branch(x, inputs, n_labels, head, name):
    
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Flatten(name='flatten_{}'.format(name))(x)

    if head['name'] == 'obd':
        hidden_size_per_unit = np.round(inputs / (n_labels - 1)).astype(int)
        layers = []
        for i in range(num_classi - 1):
            x1 = Dense(hidden_size_per_unit, name='hidden_{}_{}'.format(name,i))(x)
            x1 = LeakyReLU()(x1)
            x1 = Dropout(0.3)(x1)
            x1 = Dense(1, name='out_{}_{}'.format(name,i))(x1)
            x1 = BatchNormalization()(x1) 
            x1 = Activation('sigmoid')(x1)
            layers.append(x1)

        # stitch together
        out = Concatenate(axis=1,name="{}_output".format(name))(layers)

    else:
        # Create top layers
        x = Dropout(0.5)(x)
        x = Dense(4096, name='fc1_{}'.format(name))(x)
        x = Activation('relu')(x)
        x = Dense(4096, name='fc2_{}'.format(name))(x)
        x = Activation('relu')(x)

        if head['name'] == 'clm':
            x = Dense(1, dtype='float32')(x)
            x = BatchNormalization()(x)
            out_name = "{}_output".format(name)
            out = CLM(num_classes=n_labels, link_function=clm['link'], name = out_name, min_distance=clm['min_distance'],
                    use_slope=clm['use_slope'], fixed_thresholds=clm['fixed_thresholds'])(x)
        else:
            x = Dense(n_labels, dtype='float32')(x)
            x = BatchNormalization()(x)
            out = Activation('softmax')(x)
            out._name = "{}_output".format(name)

    return out


def create_hier_model(img_shape: np.ndarray, n_labels_macro: int, n_labels_micro: int, trainable_convs: bool, shared_layers: str,
        model_head: dict) -> Model:
    
    # Get the VGG16 pretrained model and set layers trainable status
    vgg16_conv = VGG16(include_top=False, weights='imagenet',
                       input_shape=(img_shape[0], img_shape[1], 3))
    if not trainable_convs:
        for layer in vgg16_conv.layers[:-1]:
            layer.trainable = False

    if shared_layers == '2ConvBlocks':
        
        print('2ConvBlocks shared')
        layer_name = 'block2_pool'
        x = vgg16_conv.get_layer(layer_name).output

        MacroBranch = build_deep_branch(x,4096, n_labels_macro, model_head, name='macro')
        MicroBranch = build_deep_branch(x,4096, n_labels_micro, model_head, name='micro')

    else:

        print('All Conv shared')
        x = vgg16_conv.output

        MacroBranch = build_standard_branch(x,4096, n_labels_macro, model_head, name='macro')
        MicroBranch = build_standard_branch(x,4096, n_labels_micro, model_head, name='micro')

    # Create full model
    model = Model(inputs= vgg16_conv.input, outputs=[MacroBranch,MicroBranch])
    
    return model    


def run_hier_cnn(train_data: 'tuple[np.ndarray, np.ndarray]', validation_data: 'tuple[np.ndarray, np.ndarray]',
            test_data: 'tuple[np.ndarray, np.ndarray]', optimiser_params: dict, clm: dict, obd: dict,
            loss_config: dict, loss_config2: dict, shared_layers: str, augment: bool = True,
            trainable_convs: bool = False, labels = None, return_labels = True, seed=1):
    
    # Get data X and y
    # y contains only an integer representing the class number
    X_train, y_train = train_data
    X_val, y_val = validation_data
    X_test, y_test = test_data

    # Get image shape from data
    img_shape = X_train.shape[1:3]

    # If labels param is not set, get labels from data
    if labels is None:
        labels = np.unique(y_train[:,0])
        labels_macro = np.unique(y_train[:,1])
        labels_micro = np.unique(y_train[:,2])

    n_labels = len(labels)
    n_labels_macro = len(labels_macro)
    n_labels_micro = len(labels_micro)

    if obd['enabled']:
        # Convert target labels to binary
        target_class_macro_np, target_class_macro_tf = binary_conv(n_labels_macro)
        target_class_micro_np, target_class_micro_tf = binary_conv(n_labels_micro)

        #y_val_macro_tf = tf.gather(target_class_macro_tf, y_val[:,1])
        #y_val_micro_tf = tf.gather(target_class_micro_tf, y_val[:,2])
        #y_validation = [y_val_macro_tf,y_val_micro_tf]

        #y_validation = [y_val[:,1],y_val[:,2]]
        y_validation = [y_val[:,1],np.column_stack((y_val[:,1], y_val[:,2]))]

        # Create data augmentation generator
        generator = batch_generator(X_train, y_train, optimiser_params['bs'], img_shape, 
                                    augment, seed=seed) #target_class_macro_tf, target_class_micro_tf, 

        #Define losses
        loss_macro = ordinal_distance_loss(loss_config['type'],n_labels_macro)
        loss_micro = ordinal_distance_loss_hier(loss_config2['type'],n_labels_micro)
        metric = 'mae'
        model_head = obd
    else:
        # Convert labels to one-hot encoding
        #y_validation = [to_categorical(y_val[:,1]), to_categorical(y_val[:,2])]
        y_validation = [y_val[:,1],np.column_stack((y_val[:,1], y_val[:,2]))]
        
        # Create data augmentation generator
        generator = batch_generator(X_train, y_train, optimiser_params['bs'], img_shape, augment, seed=seed)
        metric = CohenKappa(n_labels, weightage='quadratic', name='qwk')
        if clm['enabled']:
            #Define losses
            loss_macro = qwk_loss(make_cost_matrix(n_labels_macro),n_labels_macro)
            loss_micro = qwk_loss_hier(make_cost_matrix(n_labels_micro),n_labels_micro)
            model_head = clm
        else:
            loss_macro = 'categorical_crossentropy'
            loss_micro = 'categorical_crossentropy'
            model_head = None

    # Create vgg16 full model
    model = create_hier_model(img_shape, n_labels_macro, n_labels_micro, trainable_convs, shared_layers, model_head)
    
    # Define the callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    callbacks = [early_stopping]

    # Define the optimiser, the loss and compile the model
    optimiser = Adam(lr=optimiser_params['lr'])

    losses = {"macro_output":loss_macro, "micro_output":loss_micro}
    lossWeights = {"macro_output":loss_config['weight'], "micro_output":loss_config2['weight']}
        
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimiser, metrics =[metric])#,run_eagerly = True) 

    model.summary()

    # Train the model
    history: History = model.fit(
        generator,
        epochs=optimiser_params['epochs'],
        steps_per_epoch=generator.__len__(),
        validation_data=(X_val, {"macro_output":y_validation[0],"micro_output":y_validation[1]}),
        verbose=1,
        #class_weight={"macro_output":class_weights_macro,"micro_output":class_weights_micro},
        callbacks=callbacks
    )

    # Predict on test
    y_test_pred = model.predict(X_test)

    if obd['enabled']:
        distances_macro = cdist(y_test_pred[0], target_class_macro_np, metric='euclidean')
        distances_micro = cdist(y_test_pred[1], target_class_micro_np, metric='euclidean')

        labels_macro = distances_macro.argmin(axis=1)
        probas_macro = softmax(-distances_macro, axis=1)

        labels_micro = distances_micro.argmin(axis=1)
        probas_micro = softmax(-distances_micro, axis=1)
    
    else:    
        labels_macro = np.argmax(y_test_pred[0], axis=1)
        probas_macro = y_test_pred[0]

        labels_micro = np.argmax(y_test_pred[1], axis=1)
        probas_micro = y_test_pred[1]
    
    if return_labels:
        # Return predictions
        return labels_macro, labels_micro
    else:
        # return probabilities
        return probas_macro, probas_micro