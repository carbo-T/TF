# -*- coding: utf8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import mnist_inference
import mnist_train
from numpy.random import RandomState
import os

# generate new random dataset for test in 3 secs after close figure window manually
EVAL_INTERVAL_SECS = 3
NUMBER_OF_SAMPLES = 36
FIG_ROWS = 3


# display images and recognition result rather than accuracy diagram
def evaluation(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='input-x')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='input-y')

        # move sample picking into each cycle
        # rdm = RandomState(int(time.time()))
        # sample_index = rdm.randint(0, mnist.validation.num_examples)
        # validation_feed = {
        #     x: mnist.validation.images[sample_index:sample_index + 6],
        #     y_: mnist.validation.labels[sample_index:sample_index + 6]}

        # replace accuracy with actual recognition result
        y = mnist_inference.inference(x, None)
        indices = tf.argmax(y, 1)
        correct_indices = tf.argmax(y_, 1)
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            # configure TF to allocate mem properly, rather than consume all GPU mem
            config = tf.ConfigProto(allow_soft_placement=True)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    rdm = RandomState(int(time.time()))
                    sample_index = rdm.randint(0, mnist.validation.num_examples - NUMBER_OF_SAMPLES)
                    validation_feed = {
                        x: mnist.validation.images[sample_index:sample_index + NUMBER_OF_SAMPLES],
                        y_: mnist.validation.labels[sample_index:sample_index + NUMBER_OF_SAMPLES]}

                    # get global step from file name
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    indices_score, correct_indices_score = sess.run(
                        [indices, correct_indices], feed_dict=validation_feed)
                    # accuracy_score = sess.run(accuracy, feed_dict=validation_feed)
                    # print "after %s training step(s), validation accuracy = %g" % (global_step, accuracy_score)
                    print("after %s training step(s), validation result = \n%s\n, correct answer: \n%s" \
                          % (global_step, indices_score, correct_indices_score))
                    fig = plt.figure(1)
                    fig.set_size_inches(15,6)
                    for n in range(1, NUMBER_OF_SAMPLES + 1):
                        fig.add_subplot(FIG_ROWS, (NUMBER_OF_SAMPLES / FIG_ROWS + 1), n)
                        plt.title("predict: [%s]\nanswer: [%s]"
                                  % (indices_score[n - 1], correct_indices_score[n - 1]))
                        plt.imshow(mnist.validation.images[sample_index + n - 1].reshape(28, 28))
                    # fig.add_subplot(2, 3, 1)
                    # plt.imshow(mnist.validation.images[sample_index].reshape(28, 28))
                    # fig.add_subplot(2, 3, 2)
                    # plt.imshow(mnist.validation.images[sample_index + 1].reshape(28, 28))
                    # fig.add_subplot(2, 3, 3)
                    # plt.imshow(mnist.validation.images[sample_index + 2].reshape(28, 28))
                    # fig.add_subplot(2, 3, 4)
                    # plt.imshow(mnist.validation.images[sample_index + 3].reshape(28, 28))
                    # fig.add_subplot(2, 3, 5)
                    # plt.imshow(mnist.validation.images[sample_index + 4].reshape(28, 28))
                    # fig.add_subplot(2, 3, 6)
                    # plt.imshow(mnist.validation.images[sample_index + 5].reshape(28, 28))
                    plt.subplots_adjust(
                        top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.55)
                    try:
                        os.mkdir('images/')
                    except:
                        print("directory already exist")
                    plt.savefig('images/mnist_result_evaluation.jpg', format='jpg')
                    plt.show()

                else:
                    print("no checkpoint file found")
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    evaluation(mnist)


if __name__ == '__main__':
    tf.app.run()
