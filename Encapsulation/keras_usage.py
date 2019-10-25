# -*- coding:utf-8 -*-
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

from keras.layers import Embedding, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence

import os


def mnist_test():
    num_classes = 10
    img_rows, img_cols = 28, 28

    (trainX, trainY), (testX, testY) = mnist.load_data()

    # 不同底层 tf/MXNet 对输入要求不同, 根据输入图像编码格式设置输入层格式
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # int to float
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0

    # one-hot encode
    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY, num_classes)

    # create layers container
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(
        MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(
        MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # define loss, optimizer and analyze method
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))

    score = model.evaluate(testX, testY)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


# mnist_test()


def emotion_recognition():
    # maximum words to use
    max_features = 20000
    # truncate length
    maxlen = 80
    batch_size = 32
    # 25000 train data, 25000 test data
    (trainX, trainY), (testX, testY) = imdb.load_data(path="/home/youchen/PycharmProjects/TF/Encapsulation/imdb.npz", num_words=max_features)
    print(len(trainX), ' train sequences')
    print(len(testX), ' test sequences')

    # trim to the same length
    trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
    testX = sequence.pad_sequences(testX, maxlen=maxlen)
    print('x_train_shape: ', trainX.shape)
    print('x_test_shape: ', testX.shape)

    if not os.path.exists('emotion_model.h5'):
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
    else:
        model = load_model('emotion_model.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, batch_size=batch_size, epochs=1, validation_data=(testX, testY))
    model.save('emotion_model.h5')

    score = model.evaluate(testX, testY, batch_size=batch_size)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


emotion_recognition()
