# -*- coding:utf-8 -*-

import keras
from keras.datasets import mnist
from tflearn.layers.core import fully_connected
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model


def self_defined_fc():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0], 784)
    testX = testX.reshape(testX.shape[0], 784)
    # int to float
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0
    # one-hot encode
    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)

    # define input
    inputs = Input(shape=(784,))
    x = Dense(500, activation='relu')(inputs)
    predictions = Dense(10, activation='softmax')(x)

    # define model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))

    score = model.evaluate(testX, testY)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def inception_implement():
    input_img = Input(shape=(256, 256, 3))
    # branch 1
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
    # branch 2
    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
    # branch 3
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)


#   input1(784)
#       |
#       x(1)       input2(10)
#       |     \     |
#   output1(10)    output2(10)
def multi_io_structure():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0], 784)
    testX = testX.reshape(testX.shape[0], 784)
    # int to float
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0
    # one-hot encode
    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)

    input1 = Input(shape=(784,), name="input1")
    input2 = Input(shape=(10,), name="input2")

    x = Dense(1, activation='relu')(input1)
    output1 = Dense(10, activation='softmax', name="output1")(x)
    y = keras.layers.concatenate([x, input2])
    output2 = Dense(10, activation='softmax', name="output2")(y)

    model = Model(inputs=[input1, input2], outputs=[output1, output2])

    loss = {'output1': 'binary_crossentropy', 'output2': 'binary_crossentropy'}
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), loss_weights=[1, 0.1],
                  metrics=['accuracy'])
    # 用字典可避免输入顺序不一致, 当然也可以用list     [trainX, trainY], [trainY, trainY]
    model.fit(
        {'input1': trainX, 'input2': trainY},
        {'output1': trainY, 'output2': trainY},
        batch_size=128,
        epochs=20,
        validation_data=(
            [testX, testY], [testY, testY]
        )
    )


# self_defined_fc()
multi_io_structure()
