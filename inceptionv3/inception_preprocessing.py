# -*- utf-8 -*-

import glob
import os.path
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

INPUT_DATA = "../../dataset/flower_photos"

OUTPUT_FILE = "preprocess/flower_processed_data.npy"

# test and validation ratio
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_percentage, validation_percentage):
    # '../../dataset/flower_photos', '../../dataset/flower_photos/daisy', '../../dataset/flower_photos/tulips',
    # '../../dataset/flower_photos/dandelion', '../../dataset/flower_photos/sunflowers',
    # '../../dataset/flower_photos/roses']
    subdirs = [x[0] for x in os.walk(INPUT_DATA)]
    # print(subdirs)
    is_root_dir = True

    count = 0
    # init datasets
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # read all subdirs
    for sub_dir in subdirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # print(dir_name)
        for extension in extensions:
            # find all images in sub_dir
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue

        # deal with images
        for file_name in file_list:
            print(str(current_label) + file_name + "\t\t" + str(count))
            count += 1
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # split dataset randomly
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (validation_percentage + testing_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    return np.asarray(
        [training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)

        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()

    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # b = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    # state = np.random.get_state()
    # np.random.shuffle(a)
    # np.random.set_state(state)
    # np.random.shuffle(b)
    # print(a)
    # print(b)
