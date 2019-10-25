# -*- coding:utf-8 -*-

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(data_dir='../MNIST_data', one_hot=True)
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
net = input_data(shape=[None, 28, 28, 1], name='input')
net = conv_2d(net, 6, 5, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 16, 5, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 500, activation='relu')
net = fully_connected(net, 10, activation='relu')

net = regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=20, validation_set=([testX, testY]), show_metric=True)
