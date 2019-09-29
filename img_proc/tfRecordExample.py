import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time


# tfRecord defined by tf.train.Example (Protocol Buffer)
# message Example{ Features features=1;}
# message Features{ Map<string, Feature> feature=1 }
# message Feature{oneof kind{ BytesList bytes_list=1; FloatList float_list=1; Int64List int64_list=1;}}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_mnist_record(dataset=0, output_filename="record/output_mnist.tfrecords"):
    mnist = input_data.read_data_sets("../MNIST_data", dtype=tf.uint8, one_hot=True)
    images = []
    labels = []
    num_examples=0
    if dataset == 0:
        images = mnist.train.images
        labels = mnist.train.labels
        num_examples = mnist.train.num_examples
    elif dataset == 1:
        images = mnist.validation.images
        labels = mnist.validation.labels
        num_examples = mnist.validation.num_examples
    elif dataset == 2:
        images = mnist.test.images
        labels = mnist.test.labels
        num_examples = mnist.test.num_examples
    print(num_examples)
    # define resolution
    # pixels = images.shape[1]
    # print(images[0].shape)

    writer = tf.python_io.TFRecordWriter(output_filename)
    for index in range(num_examples):
        # convert img to str
        image_raw = images[index].tostring()
        # create Example Protocol Buffer
        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'pixels': _int64_feature(pixels),
        #     'label': _int64_feature(np.argmax(labels[index])),
        #     'image_raw': _bytes_feature(image_raw)
        # }))
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_raw),
            'label': _int64_feature(np.argmax(labels[index])),
            'height': _int64_feature(28),
            'width': _int64_feature(28),
            'channels': _int64_feature(1),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_mnist_record(input_filename="output_mnist.tfrecords"):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([input_filename])
    # read an example
    _, serialized_example = reader.read(filename_queue)
    # resolve the example
    features = tf.parse_single_example(
        serialized_example,
        features={
            # tf.FixedLenFeature return a Tensor
            # tf.VarLenFeature return a SparseTensor
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    # convert from str to img
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        img, answer, num_pixels = sess.run([image, label, pixels])
        print("answer: %d, num of pixels: %d" % (answer, num_pixels))
        plt.imshow(img.reshape(28, 28))
        plt.show()
        time.sleep(3)


def main():
    save_mnist_record(0, "record/mnist_train.tfrecord")
    save_mnist_record(1, "record/mnist_validation.tfrecord")
    save_mnist_record(2, "record/mnist_test.tfrecord")
    # read_mnist_record()


if __name__ == '__main__':
    main()
