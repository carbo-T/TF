# -*- coding: utf8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import mnist_inference

# define input, output, batch and training params

BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
score_filename = "accuracy_score_cnn.txt"


# train a convolutional neural network
def train(mnist, continue_train=False):
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # moving average, cross entropy, loss function with regularization and learning rate
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # initialize persistence class
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if continue_train:
            ckpt = tf.train.get_checkpoint_state(
                MODEL_SAVE_PATH
            )
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        # create directory
        try:
            os.mkdir(MODEL_SAVE_PATH)
        except:
            print("directory already exist")

        # define accuracy
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_result = list(range(int(TRAINING_STEPS / 1000)))

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            txs = mnist.test.images[0:BATCH_SIZE]
            test_feed = {
                x: np.reshape(txs, (BATCH_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS)),
                y_: mnist.test.labels[0:BATCH_SIZE]}

            accuracy_score = sess.run(accuracy, feed_dict=test_feed)
            test_result[int(i / 1000)] = accuracy_score

            if i % 1000 == 0:
                print("after %d training step(s), loss on training batch is %g , validation accuracy = %g" % (
                    step, loss_value, accuracy_score))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step
                )

    # np.savetxt(score_filename, test_result, fmt="%0.4f")
    #
    # dispImg(test_result, 'accuracy_score')
    # plt.show()


def dispImg(test_result, filename):
    # draw a graph of accuracy using matplotlib
    iteration_count = range(0, TRAINING_STEPS, 1000)
    plt.figure(num=1, figsize=(15, 8))
    plt.title("Plot accuracy", size=20)
    plt.xlabel("iteration count", size=14)
    plt.ylabel("accuracy/%", size=14)
    test_note = [TRAINING_STEPS - 1000, test_result[TRAINING_STEPS / 1000 - 1]]
    plt.annotate('test-' + str(test_note), xy=(test_note[0], test_note[1]),
                 xytext=(test_note[0] + 1000, test_note[1] - 0.07), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid(True)
    plt.plot(iteration_count, test_result, linestyle='-.', marker='X', label='test data')
    plt.legend(loc="upper left")
    try:
        os.mkdir('images/')
    except:
        print("directory already exist")
    plt.savefig('images/%s.png' % filename, format='png')


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    print("start")
    train(mnist, True)


if __name__ == '__main__':
    tf.app.run()
