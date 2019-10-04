# -*- coding: utf8 -*-
import tensorflow as tf

# define basic params
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEPTH = 6
CONV1_SIZE = 5

CONV2_DEPTH = 16
CONV2_SIZE = 5

FC_SIZE = 84


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)

        # calc mean
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # calc standard deviation
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def inference(input_tensor, train, regularizer):
    # print(input_tensor.get_shape())
    # define layer1 forward propagation
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEPTH],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable("bias", [CONV1_DEPTH], initializer=tf.constant_initializer(0.0))
        # strides 中间两项表示长宽方向步长1
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        variable_summaries(conv1_weights, 'layer1-conv1' + '/weights')
        variable_summaries(conv1_biases, 'layer1-conv1' + '/biases')
    # define layer2 forward propagation, max pooling, size 2*2, step 2*2, all 0 filling
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(pool1.get_shape())
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable("bias", [CONV2_DEPTH], initializer=tf.constant_initializer(0.0))
        # size 5*5, depth 64, step 1, all 0 filling
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        variable_summaries(conv2_weights, 'layer3-conv2' + '/weights')
        variable_summaries(conv2_biases, 'layer3-conv2' + '/biases')
    with tf.variable_scope('layer4-poll2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(pool2.get_shape())
    # pool_shape[0] means the num of data from a batch, get_shape->[num, width, height, depth]
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [tf.shape(pool2)[0], nodes])
    # print(reshaped.get_shape())
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            'weights',
            [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # fc layer regularize
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

        variable_summaries(fc1_weights, 'layer5-fc1' + '/weights')
        variable_summaries(fc1_biases, 'layer5-fc1' + '/biases')
    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable(
            'weight',
            [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weight))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weight) + fc2_biases

        variable_summaries(fc2_weight, 'layer6-fc2' + '/weights')
        variable_summaries(fc2_biases, 'layer6-fc2' + '/biases')
    return logit
