# -*- utf-8 -*-
import glob
import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = 'preprocess/flower_processed_data.npy'

TRAIN_FILE = 'model/'

CKPT_FILE = '../../dataset/inception_v3.ckpt'

# params
LEARNING_RATE = 0.0001
STEPS = 1000
BATCH = 32
N_CLASSES = 5

# lasers don't load from ckpt, i.e. the last fc layer
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

TRAINING = False


flower_label = ["daisy雏菊", "roses玫瑰", "tulips郁金香", "sunflowers向日葵", "dandelion蒲公英"]


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    # enumerate params in v3 model, check if it need to be loaded
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main():
    # processed_data = np.load("preprocess/test_flower.npy", allow_pickle=True)
    # test_images = processed_data[0]
    # test_labels = processed_data[1]

    # load preprocessed data
    processed_data = np.load(INPUT_DATA, allow_pickle=True)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    # np.save("preprocess/training_flower.npy", np.asarray([training_images, training_labels]))
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    # np.save("preprocess/validation_flower.npy", np.asarray([validation_images, validation_labels]))
    test_images = processed_data[4]
    test_labels = processed_data[5]
    # np.save("preprocess/test_flower.npy", np.asarray([test_images, test_labels]))

    print("%d training examples, %d validation examples and %d testing examples." % (
        n_training_example, len(validation_labels), len(test_labels)))

    # define inputs
    images = tf.placeholder(
        tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # define model
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES, is_training=False)
    # get trainable variable
    trainable_variables = get_trainable_variables()
    # define cross entropy
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # calc accuracy
    with tf.name_scope('evaluation'):
        prediction = tf.argmax(logits, 1)
        correct_answer = labels
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # define func to load model
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True
    )

    # define saver
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # init
        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(
            TRAIN_FILE
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # load origin model
            print('loading tuned variables from %s' % CKPT_FILE)
            load_fn(sess)

        start = 0
        end = BATCH
        if TRAINING:
            for i in range(STEPS):
                sess.run(train_step, feed_dict={
                    images: training_images[start:end],
                    labels: training_labels[start:end]
                })

                if i % 20 == 0 or i + 1 == STEPS:
                    saver.save(sess, TRAIN_FILE, global_step=i)
                    validation_accuracy = sess.run(evaluation_step, feed_dict={
                        images: validation_images,
                        labels: validation_labels
                    })
                    print('step %d: validation accuracy = %.1f%%' % (i, validation_accuracy * 100.0))

                start = end
                if start == n_training_example:
                    start = 0

                end = start + BATCH
                if end > n_training_example:
                    end = n_training_example

            # test accuracy
            test_acccuracy = sess.run(evaluation_step, feed_dict={
                images: test_images,
                labels: test_labels
            })
            print('final test accuracy = %.1f%%' % (test_acccuracy * 100.0))
        else:
            while True:
                index = np.random.randint(0, len(test_labels) - 2)
                # test accuracy
                prediction_score, correct_answer_score = sess.run([prediction, correct_answer], feed_dict={
                    images: test_images[index:index+1],
                    labels: test_labels[index:index+1]
                })
                result = [(flower_label[x]+str(x)) for x in prediction_score]
                answer = [(flower_label[x]+str(x)) for x in correct_answer_score]
                # print(result)
                # print(answer)
                plt.imshow(test_images[index])
                print('test result: %s, correct answer: %s' % (
                    result, answer))
                plt.show()
                time.sleep(3)


if __name__ == '__main__':
    main()
