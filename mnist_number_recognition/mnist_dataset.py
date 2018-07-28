import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
import numpy as np

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# print "basic information of mnist dataset"
# print "mnist training data size: ", mnist.train.num_examples
# print "mnist validating data size: ", mnist.validation.num_examples
# print "mnist testing data size: ", mnist.test.num_examples
# print "mnist example training data: ", mnist.train.images[0]
# print "mnist example training data label", mnist.train.labels[0]

# define input and output data size
INPUT_NODE = 784
OUTPUT_NODE = 10

# params for neural network
LAYER1_NODE = 500
BATCH_SIZE = 1000
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.999
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99


# calc the result of forward propagation,
# ***original method***
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # use current value when there's no moving average model
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# training process
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    # generate hidden layer params
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # generate output layer params
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # forward propagation
    y = inference(x, None, weight1, biases1, weight2, biases2)

    # used to store training cycles
    global_step = tf.Variable(0, trainable=False)

    # define EMA function to increase robustness when predict
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # # forward propagation with moving average function
    # average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    # calc cross_entropy mean for current batch
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # calc L2 regularization loss function
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)
    loss = cross_entropy_mean + regularization

    # learning rate = learning rate * LEARNING_RATE_DECAY ^ (global_step / decay_step)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # combine backward propagation and EMA value modification
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # prepare validation dataset to stop optimization
        validation_feed = {x: mnist.validation.images,
                           y_: mnist.validation.labels}

        # define test dataset for final evaluation
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        validation_result = range(TRAINING_STEPS / 1000)
        test_result = range(TRAINING_STEPS / 1000)
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # print "average_y: ", average_y, sess.run(average_y, feed_dict=validation_feed)
                # print "y_: ", y_, sess.run(y_, feed_dict=validation_feed)

                validate_acc = sess.run(accuracy, feed_dict=validation_feed)
                validation_result[i / 1000] = validate_acc
                # print "after %d training step(s), validation accuracy using average model is %g " % (i, validate_acc)

                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

                test_acc = sess.run(accuracy, feed_dict=test_feed)
                test_result[i / 1000] = test_acc
                # print "after %d training step(s), test accuracy using average model is %g " % (i, test_acc)

        print validation_result
        print test_result

    # draw a graph of accuracy using matplotlib
    iteration_count = range(0, TRAINING_STEPS, 1000)
    plt.figure(num=1, figsize=(15, 8))
    plt.title("Plot accuracy", size=20)
    plt.xlabel("iteration count", size=14)
    plt.ylabel("accuracy/%", size=14)
    validation_note = [TRAINING_STEPS - 1000, validation_result[TRAINING_STEPS / 1000 - 1]]
    test_note = [TRAINING_STEPS - 1000, test_result[TRAINING_STEPS / 1000 - 1]]
    plt.annotate('validate-' + str(validation_note), xy=(test_note[0], test_note[1]),
                 xytext=(test_note[0] - 1000, test_note[1] - 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('test-' + str(test_note), xy=(test_note[0], test_note[1]),
                 xytext=(test_note[0] + 1000, test_note[1] - 0.07), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid(True)
    plt.plot(iteration_count, validation_result, color='b', linestyle='-', marker='o', label='validation data')
    plt.plot(iteration_count, test_result, linestyle='-.', marker='X', label='test data')
    plt.legend(loc="upper left")
    try:
        os.mkdir('images/')
    except:
        print("directory already exist")
    plt.savefig('images/mnist_accuracy_evaluation.png', format='png')
    img_vector = mnist.train.images[5]
    img_length = int(np.sqrt(INPUT_NODE))
    img = np.ndarray([img_length, img_length])
    # print "image size: ", img_length, "*", img_length
    for c in range(INPUT_NODE):
        # print "image indices: ", c / img_length, "*", c % img_length
        img[c / img_length][c % img_length] = img_vector[c]
    plt.figure(num=2, figsize=(15, 8))
    plt.imshow(img)
    plt.show()


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print "basic information of mnist dataset"
    print "mnist training data size: ", mnist.train.num_examples
    print "mnist validating data size: ", mnist.validation.num_examples
    print "mnist testing data size: ", mnist.test.num_examples
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
