import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from scipy.special import softmax, gamma
from scipy.stats import binom, poisson
from scipy.integrate import simps
from tensorflow import map_fn, argmax, gather
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cdist

#### BETA DISTRIBUTION ####

# Compute the beta probability function (a,b) for value x.
def beta(x, a, b):
    return gamma(a+b) / (gamma(a) * gamma(b)) * x ** (a-1) * (1 - x) ** (b-1)

def beta_mean(a, b):
    return a / (a + b)

def beta_var(a, b):
    return (a * b) / ((a + b + 1) * (a + b) ** 2)

def find_beta_params(desired_mean, min_var, max_var, a_range=100, b_range=100):
    for a in range(1, a_range + 1):
        for b in range(1, b_range + 1):
            mean = beta_mean(a, b)
            var = beta_var(a, b)

            if abs(desired_mean - mean) < 0.002 and var >= min_var and var <= max_var:
                return a, b, mean, var

def find_beta_params_for_classes(n):
    means = np.linspace(0, 1, n + 2)[1:-1]
    params = []

    for mean in means:
        params.append(find_beta_params(mean, 0.0005, 0.005))
    
    return params


# Get n evenly-spaced intervals in [0,1].
def get_intervals(n):
    points = np.linspace(0, 1.0, n + 1)
    intervals = []
    for i in range(0, points.size - 1):
        intervals.append((points[i], points[i+1]))

    return intervals

# Get probabilities from beta distribution (a,b) for n splits
def get_beta_probabilities(n, a, b):
    intervals = get_intervals(n)
    probs = []

    for interval in intervals:
        x = np.arange(interval[0], interval[1], 1e-6)
        y = beta(x, a, b)
        probs.append(simps(y,x))

    return probs


# Compute categorical cross-entropy applying regularization based on beta distribution to targets.
def categorical_ce_beta_regularized(num_classes, eta=1.0):
    # Params [a,b] for beta distribution
    params = {}

    params['3'] = [
        [1,4],
        [4,4],
        [4,1]
    ]

    params['4'] = [
        [1,6],
        [6,10],
        [10,6],
        [6,1]
    ]

    params['5'] = [
        [1,8],
        [6,14],
        [12,12],
        [14,6],
        [8,1]
    ]
    params['6'] = [
        [1,10],
        [7,20],
        [15,20],
        [20,15],
        [20,7],
        [10,1]
    ]

    params['8'] = [
        [1,14],
        [7,31],
        [17,37],
        [27,35],
        [35,27],
        [37,17],
        [31,7],
        [14,1]
    ]

    params['10'] = [
        [1, 18],
        [8, 45],
        [19, 57],
        [32, 59],
        [45, 55],
        [55, 45],
        [59, 32],
        [57, 19],
        [45, 8],
        [18, 1]
    ]

    # Precompute class probabilities for each label
    cls_probs = []
    for i in range(0, num_classes):
        cls_probs.append(get_beta_probabilities(num_classes, params[str(num_classes)][i][0], params[str(num_classes)][i][1]))

    def _compute(y_true, y_pred):
        y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y), axis=-1), tf.float32), y_true)
        y_true = (1 - eta) * y_true + eta * y_prob

        return categorical_crossentropy(y_true, y_pred)

    return _compute

#### EXPONENTIAL DISTRIBUTION ####
def get_exponential_probabilities(n, tau=1.0, l=1.0):
    probs = []

    for true_class in range(0, n):
        probs.append(-(np.abs(np.arange(0, n) - true_class) / tau)**l)

    return softmax(np.array(probs), axis=1)


# Compute categorical cross-entropy applying regularization based on exponential distribution to targets.
def categorical_ce_exponential_regularized(num_classes, eta=1.0, tau=1.0, l=1.0):
    cls_probs = get_exponential_probabilities(num_classes, tau, l)

    def _compute(y_true, y_pred):
        y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
        y_true = (1 - eta) * y_true + eta * y_prob

        return categorical_crossentropy(y_true, y_pred)

    return _compute


#### POISSON DISTRIBUTION ####

# Get probabilities for each true class based on a poisson distribution
# n is the number of classes
# returns a matrix where each row represents the true class and each column the probability for class n
def get_poisson_probabilities(n):
    probs = []

    for true_class in range(1, n+1):
        probs.append(poisson.pmf(np.arange(0, n), true_class))

    return softmax(np.array(probs), axis=1)

# Compute categorical cross-entropy applying regularization based on poisson distribution to targets.
def categorical_ce_poisson_regularized(num_classes, eta=1.0):
    cls_probs = get_poisson_probabilities(num_classes)

    def _compute(y_true, y_pred):
        y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
        y_true = (1 - eta) * y_true + eta * y_prob

        return categorical_crossentropy(y_true, y_pred)

    return _compute


#### BINOMINAL DISTRIBUTION ####


def get_binominal_probabilities(n):
    params = {}
    
    params['4'] = [
        0.2,
        0.4,
        0.6,
        0.8
    ]

    params['5'] = [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9
    ]

    params['6'] = [
        0.1,
        0.26,
        0.42,
        0.58,
        0.74,
        0.9
    ]

    params['8'] = [
        0.1,
        0.21428571,
        0.32857143,
        0.44285714,
        0.55714286,
        0.67142857,
        0.78571429,
        0.9
    ]

    params['10'] = [
        0.1,
        0.18888889,
        0.27777778,
        0.36666667,
        0.45555556,
        0.54444444,
        0.63333333,
        0.72222222,
        0.81111111,
        0.9
    ]


    probs = []

    for true_class in range(0, n):
        probs.append(binom.pmf(np.arange(0, n), n - 1, params[str(n)][true_class]))

    return np.array(probs)

# Compute categorical cross-entropy applying regularization based on binomial distribution to targets.
def categorical_ce_binomial_regularized(num_classes, eta=1.0):
    cls_probs = get_binominal_probabilities(num_classes)

    def _compute(y_true, y_pred):
        y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
        y_true = (1 - eta) * y_true + eta * y_prob

        return categorical_crossentropy(y_true, y_pred)

    return _compute


def make_cost_matrix(num_ratings):
    """
    Create a quadratic cost matrix of num_ratings x num_ratings elements.

    :param thresholds_b: threshold b1.
    :param thresholds_a: thresholds alphas vector.
    :param num_labels: number of labels.
    :return: cost matrix.
    """

    cost_matrix = np.reshape(np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings))
    cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_ratings - 1) ** 2.0
    return np.float32(cost_matrix)

def qwk_loss_base(cost_matrix):
    """
    Compute QWK loss function.

    :param pred_prob: predict probabilities tensor.
    :param true_prob: true probabilities tensor.
    :param cost_matrix: cost matrix.
    :return: QWK loss value.
    """
    def _qwk_loss_base(true_prob, pred_prob):

        targets = tf.argmax(true_prob, axis=1)
        costs = tf.gather(cost_matrix, targets)

        numerator = costs * pred_prob
        numerator = tf.reduce_sum(numerator)

        sum_prob = tf.reduce_sum(pred_prob, axis=0)
        n = tf.reduce_sum(true_prob, axis=0)

        a = tf.reshape(tf.matmul(cost_matrix, tf.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
        b = tf.reshape(n / tf.reduce_sum(n), shape=[-1])
        
        epsilon = 10e-9

        denominator = a * b
        denominator = tf.reduce_sum(denominator) + epsilon

        return numerator / denominator

    return _qwk_loss_base


def qwk_loss(cost_matrix,num_classes):
    """
    Compute QWK loss function.

    :param pred_prob: predict probabilities tensor.
    :param true_prob: true probabilities tensor.
    :param cost_matrix: cost matrix.
    :return: QWK loss value.
    """
    def _qwk_loss(true_prob, pred_prob):

        true_prob = tf.squeeze(tf.one_hot(true_prob,num_classes),axis=1)

        targets = tf.argmax(true_prob, axis=1)
        costs = tf.gather(cost_matrix, targets)

        numerator = costs * pred_prob
        numerator = tf.reduce_sum(numerator)

        sum_prob = tf.reduce_sum(pred_prob, axis=0)
        n = tf.reduce_sum(true_prob, axis=0)

        a = tf.reshape(tf.matmul(cost_matrix, tf.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
        b = tf.reshape(n / tf.reduce_sum(n), shape=[-1])

        epsilon = 10e-9

        denominator = a * b
        denominator = tf.reduce_sum(denominator) + epsilon

        return numerator / denominator

    return _qwk_loss

def ordinal_distance_loss_base(loss,n_classes,class_weight):
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = tf.convert_to_tensor(target_class, dtype=tf.float32)
    if loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    elif loss == 'mse':
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    else:
        raise ValueError('Accepted losses are mae and mse')


    def _ordinal_distance_loss_base(target, net_output):
        
        if class_weight is not None:
            loss1 = loss(target,net_output)
            class_weight1 = tf.convert_to_tensor(class_weight,dtype=tf.float32)
            weight_mask = tf.gather(class_weight1,  tf.cast(target, tf.int32))
            loss1 = tf.math.multiply(loss1, weight_mask)
        else:
            loss1 = loss(target,net_output)
        
        return loss1

    return _ordinal_distance_loss_base


def ordinal_distance_loss(loss,n_classes):
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = tf.convert_to_tensor(target_class, dtype=tf.float32)
    if loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    elif loss == 'mse':
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    else:
        raise ValueError('Accepted losses are mae and mse')


    def _ordinal_distance_loss(target, net_output):
        target = tf.gather(target_class, tf.cast(target, tf.int32))
        net_output = tf.expand_dims(net_output, axis=1)

        return loss(target,net_output)

    return _ordinal_distance_loss



def split_labels(y_true):
    child_y = tf.expand_dims(y_true[:,0],1)
    y_true = tf.expand_dims(y_true[:,1],1)
    return y_true, child_y

def get_two_splits(y_true):
    split = int(tf.shape(y_true)[1]/2)
    child_y = y_true[:,(-split):]
    y_true = y_true[:,:split]
    return y_true, child_y


def ordinal_distance_loss_hier(loss,n_classes):
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = tf.convert_to_tensor(target_class, dtype=tf.float32)
    if loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    elif loss == 'mse':
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    else:
        raise ValueError('Accepted losses are mae and mse')

    def _apply_loss(y_all):
        y_true, y_pred = get_two_splits(y_all)
        return loss(y_true, y_pred)

    def _ordinal_distance_loss_hier(target, net_output):

        n = tf.shape(target[:,1])[0]
        y_true, partitions = split_labels(target)
        partitions = tf.cast(tf.reshape(partitions ,[n, ]), tf.int32)

        y_true = tf.gather(target_class, tf.cast(y_true, tf.int32))
        y_true = tf.squeeze(y_true, axis=1)

        y_elements = tf.dynamic_partition(tf.concat([y_true, net_output], 1), partitions, 4)

        losses = [] 
        for t in y_elements:
            t = tf.expand_dims(t,axis=0)
            loss = tf.map_fn(_apply_loss, t, fn_output_signature=tf.float32)
            losses.append(loss)
        
        print(losses)
        mask = tf.cast(losses, dtype=tf.bool)
        nonzero_x = tf.boolean_mask(losses, mask)

        return tf.reduce_mean(nonzero_x, axis=0)
    
    return _ordinal_distance_loss_hier
    

def qwk_loss_hier(cost_matrix,num_classes):

    def _qwk_loss_hier(true_prob, pred_prob):   

        def _apply_qwk(y_all):
            y_true, y_pred = get_two_splits(y_all)      

            targets = tf.argmax(y_true, axis=1)
            costs = tf.gather(cost_matrix, targets)       

            numerator = costs * y_pred
            numerator = tf.reduce_sum(numerator)
            if numerator == 0:
                return 0.0
               
            sum_prob = tf.reduce_sum(y_pred, axis=0)
            n = tf.reduce_sum(y_true, axis=0)
        
            a = tf.reshape(tf.matmul(cost_matrix, tf.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
            b = tf.cast(tf.reshape(n / tf.reduce_sum(n), shape=[-1]),tf.float32)
        
            epsilon = 10e-9
        
            denominator = a * b
            denominator = tf.reduce_sum(denominator) + epsilon
        
            return numerator / denominator


        n = tf.shape(true_prob[:,1])[0]
        
        y_true, partitions = split_labels(true_prob)
        partitions = tf.cast(tf.reshape(partitions ,[n, ]), tf.int32)

        y_true = tf.squeeze(tf.one_hot(y_true,num_classes),axis=1)

        y_elements = tf.dynamic_partition(tf.concat([y_true, pred_prob], 1), partitions, 4)

        losses = [] 
        for t in y_elements:
            t = tf.expand_dims(t,axis=0)
            loss = tf.map_fn(_apply_qwk, t, fn_output_signature=tf.float32)
            losses.append(loss)

        mask = tf.cast(losses, dtype=tf.bool)
        nonzero_x = tf.boolean_mask(losses, mask)

        return tf.reduce_mean(nonzero_x, axis=0)

    return _qwk_loss_hier 