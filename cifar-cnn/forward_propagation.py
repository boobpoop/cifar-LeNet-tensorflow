import tensorflow as tf
import numpy as np

#step 2------define network structure

OUTPUT_NODE = 10
BATCH_SIZE = 100
IMAGE_SIZE = 32
INPUT_CHANNELS = 3

CONV1_SIZE = 5
CONV1_DEEP = 6
CONV2_SIZE = 5
CONV2_DEEP = 16
FC1_SIZE = 120
FC2_SIZE = 84

def get_weight(shape):
    weight = tf.get_variable("weight", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
    return weight

def get_bias(shape):
    bias = tf.get_variable("bias", shape, initializer = tf.constant_initializer(0.1))
    return bias

def conv2d(x, w):
    conv = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = "SAME")
    return conv

def max_pool(x):
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    return pool

def forward_propagation(input_data_x):

    #use LeNet-5 network structure
    with tf.variable_scope("layer1--conv1"):
        conv1_weight = get_weight([CONV1_SIZE, CONV1_SIZE, INPUT_CHANNELS, CONV1_DEEP])   
        conv1_bias = get_bias([CONV1_DEEP])
        conv1 = conv2d(input_data_x, conv1_weight)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.variable_scope("layer2--pool1"):
        pool1 = max_pool(relu1)

    with tf.variable_scope("layer3--conv2"):
        conv2_weight = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
        conv2_bias = get_bias([CONV2_DEEP])
        conv2 = conv2d(pool1, conv2_weight)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    with tf.variable_scope("layer4--pool2"):
        pool2 = max_pool(relu2)

    #dimension reduction
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1 , nodes])
    

    with tf.variable_scope("layer5--fc1"):
        fc1_weight = get_weight([nodes, FC1_SIZE])
        fc1_bias = get_bias([FC1_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)

    with tf.variable_scope("layer6--fc2"):
        fc2_weight = get_weight([FC1_SIZE, FC2_SIZE])
        fc2_bias = get_bias([FC2_SIZE])
        fc2 = tf.nn.relu(tf.matmul(fc1,  fc2_weight) + fc2_bias)

    with tf.variable_scope("layer7--fc2"):
        fc3_weight = get_weight([FC2_SIZE, OUTPUT_NODE])
        fc3_bias = get_bias([OUTPUT_NODE])
        y = tf.nn.relu(tf.matmul(fc2, fc3_weight) + fc3_bias)
    return (y, fc1_weight, fc2_weight, fc3_weight)
