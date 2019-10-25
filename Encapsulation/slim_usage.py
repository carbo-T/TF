# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])

    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, 2, scope='layer4-max-pool')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 500, scope='layer5-fc')
    net = slim.fully_connected(net, 10, scope='output')
    return net
