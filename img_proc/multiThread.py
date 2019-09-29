import tensorflow as tf
import numpy as np
import threading
import time
import os
import preprocessing
import mnist_inference
import matplotlib.pyplot as plt


# ********** queue operation ***********
def queue_op():
    # FIFOQueue & RandomShuffleQueue
    # maximum 2 int elements
    q = tf.FIFOQueue(2, "int32")

    init = q.enqueue_many(([0, 10],))

    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        init.run()
        for _ in range(5):
            # including dequeue, add 1, enqueue
            v, _ = sess.run([x, q_inc])
            # print(v)


# tf.train.Coordinator enable thread synchronization
# request_stop, should_stop, join
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d" % worker_id)
            coord.request_stop()
        else:
            time.sleep(0.5)
            print("Working on id: %d" % worker_id)
        time.sleep(1)


# coord = tf.train.Coordinator()
# threads = [
#     threading.Thread(target=MyLoop, args=(coord, i), ) for i in range(5)
# ]
# # start all threads
# for t in threads:
#     t.start()
# # wait for all threads to stop
# coord.join(threads)

# ******** tf.QueueRunner **********
def threads_mgmt():
    queue = tf.FIFOQueue(100, 'float')
    enqueue_op = queue.enqueue([tf.random_normal([1])])
    # create 5 threads
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
    # added to default collection tf.GraphKeys.QUEUE_RUNNERS,
    # start_queue_runner() will start all threads in the specified collection
    tf.train.add_queue_runner(qr)
    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(15):
            print(sess.run(out_tensor)[0])
            time.sleep(0.2)
    coord.request_stop()
    coord.join(threads)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# simulate big data situation
def generate_files():
    # how many files to write
    num_shard = 3
    # how much data in a file
    instances_per_shard = 6
    record_path = "record/"
    try:
        os.mkdir(record_path)
    except:
        print("directory already exist")

    # data 0000n-of-0000m, n means file No., m means how many files the data has been stored as
    for i in range(num_shard):

        filename = (os.path.join(record_path, "data.tfrecords-%.5d-of-%.5d" % (i, num_shard)))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instances_per_shard):
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)
            }))
            writer.write(example.SerializeToString())
        writer.close()


def read_files():
    # 获取文件列表
    record_path = "record/"
    files = tf.train.match_filenames_once(os.path.join(record_path, "data.tfrecords-*"))

    # 1 epochs means 1 cycle
    filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'i': tf.FixedLenFeature([], tf.int64),
            'j': tf.FixedLenFeature([], tf.int64),
        }
    )

    with tf.Session() as sess:
        # match_filename_once() needs to be initialized
        tf.local_variables_initializer().run()
        print(sess.run(files))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(18):
            print(sess.run([features['i'], features['j']]))
        coord.request_stop()
        coord.join(threads)

    return features


def batch_example():
    features = read_files()

    print("____ end of read files _____")

    example, label = features['i'], features['j']
    batch_size = 3
    # queue capacity, larger means more memory usage, smaller means can be blocked and less efficient
    capacity = 1000 + 3 * batch_size
    # example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
    # min_after_dequeue represent the num of data needed for dequeue operation which is blocked when the num inadequate
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,
                                                        min_after_dequeue=6)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # combine
        for i in range(6):
            curr_exp_b, curr_lab_b = sess.run([example_batch, label_batch])
            print(curr_exp_b, curr_lab_b, "lll")

        coord.request_stop()
        coord.join(threads)


# ************* use inceptionV3 data to generate data for training **************
def write_record(name, image, label):
    writer = tf.python_io.TFRecordWriter(name)
    for index in range(len(image)):
        # convert img to str
        image_raw = image[index].tobytes()
        print(label[index])
        print(image[index].shape[0])
        print(image[index].shape[1])
        print(image[index].shape[2])
        # create Example Protocol Buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_raw),
            'label': _int64_feature(label[index]),
            'height': _int64_feature(image[index].shape[0]),
            'width': _int64_feature(image[index].shape[1]),
            'channels': _int64_feature(image[index].shape[2]),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def generate_record(output_filename="output_flower.tfrecords"):
    input_data = "../inceptionv3/preprocess/validation_flower.npy"
    processed_data = np.load(input_data, allow_pickle=True)
    training_images = processed_data[0]
    training_labels = processed_data[1]

    input_data = "../inceptionv3/preprocess/test_flower.npy"
    processed_data = np.load(input_data, allow_pickle=True)
    validation_images = processed_data[0]
    validation_labels = processed_data[1]

    write_record("output_flower_train.tfrecord", training_images, training_labels)
    write_record("output_flower_validation.tfrecord", validation_images, validation_labels)

    print("training_images: " + str(len(training_labels)))
    print("validation_images: " + str(len(validation_labels)))


def read_record(file_regex="record/output_flower_*.tfrecord"):
    files = tf.train.match_filenames_once(file_regex)
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64)
        })

    image, label = features['image'], tf.cast(features['label'], tf.int32)
    height, width = tf.cast(features['height'], tf.int32), tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)

    # image decoding
    decoded_img = tf.decode_raw(image, tf.float32)
    # decoded_img.set_shape(268203)
    decoded_img = tf.reshape(decoded_img,
                             shape=[height, width, channels])
    return decoded_img, label


def tfrecord_parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64)
        })
    image, label = features['image'], tf.cast(features['label'], tf.int32)
    height, width = tf.cast(features['height'], tf.int32), tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)

    # image decoding
    decoded_img = tf.decode_raw(image, tf.uint8)
    # decoded_img.set_shape(268203)
    # decoded_img.set_shape([height, width, channels])
    decoded_img = tf.reshape(decoded_img,
                             shape=[height, width, channels])
    return decoded_img, label


# ** wrong image dtype may cause " Input to reshape is a tensor with xxx values, but the requested shape has xxx "
# such as uint8 and float32, float32 is usually used for training, whereas uint8 more likely used for image storage
# ** must have channel 3 but has channels 1 problem is caused by image preprocessing
def process_data(doTrain=True):
    image_size = 28
    num_channels = 1
    num_of_labels = 10
    min_after_dequeue = 2000
    shuffle_buffer = 10000
    num_epochs = 50  # same effect as training_rounds
    batch_size = 500
    training_rounds = 5000
    training_images = 55000  # 362
    validation_images = 5000  # 367
    test_images = 10000
    train_files = tf.train.match_filenames_once("record/mnist_train.tfrecord")
    validation_files = tf.train.match_filenames_once("record/mnist_validation.tfrecord")
    test_files = tf.train.match_filenames_once("record/mnist_test.tfrecord")

    # ********** define neural network structure and forward propagation **********
    learning_rate_base = 0.8
    learning_rate_decay = 0.99
    regularization_rate = 0.0001
    moving_average_decay = 0.99
    x = tf.placeholder(tf.float32, [None,
                                    image_size,
                                    image_size,
                                    num_channels], name='x-input')
    y_ = tf.placeholder(tf.float32, [None], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # moving average, cross entropy, loss function with regularization and learning rate
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    # calc loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.cast(y_, tf.int32))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        training_images / batch_size,
        learning_rate_decay
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # define accuracy
    prediction = tf.argmax(y, 1)
    answer = tf.cast(y_, tf.int64)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # test_result = list(range(int(training_rounds / 500)))

    # # ********** original tfrecord data operator **********
    # decoded_img, label = read_record("record/mnist_train.tfrecord")
    # # img preprocessing
    # # distorted_img = tf.image.resize_images(decoded_img, [image_size, image_size], method=0)
    # distorted_img = preprocessing.process_for_train(decoded_img, image_size, image_size, None, 1)
    # distorted_img.set_shape([image_size, image_size, num_channels])
    # # print(distorted_img.shape)
    #
    # # create batch
    # total_sample = training_images + validation_images
    # capacity = min_after_dequeue + batch_size * 3
    # image_batch, label_batch = tf.train.shuffle_batch([distorted_img, label], batch_size=batch_size,
    #                                               capacity=capacity, num_threads=64,
    #                                               min_after_dequeue=min_after_dequeue)

    # ********** tfrecord dataset **********
    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(tfrecord_parser)
    dataset = dataset.map(
        lambda image, label: (
            preprocessing.process_for_train(tf.image.convert_image_dtype(image, dtype=tf.float32), image_size,
                                            image_size, None, 1), label
        # tf.image.resize_images(tf.image.convert_image_dtype(image, dtype=tf.float32), [image_size, image_size]), label
        ))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    # match_filename_once has similar mechanism as placeholder
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()

    # ********** validation dataset **********
    validation_dataset = tf.data.TFRecordDataset(validation_files)
    validation_dataset = validation_dataset.map(tfrecord_parser).map(
        lambda image, label: (
            tf.image.resize_images(tf.image.convert_image_dtype(image, dtype=tf.float32), [image_size, image_size]),
            label
        ))
    validation_dataset = validation_dataset.batch(validation_images)
    validation_dataset = validation_dataset.repeat(None)
    validation_iterator = validation_dataset.make_initializable_iterator()
    validation_image_batch, validation_label_batch = validation_iterator.get_next()

    # ********** test dataset **********
    test_dataset = tf.data.TFRecordDataset(test_files)
    test_dataset = test_dataset.map(tfrecord_parser).map(
        lambda image, label: (
            tf.image.resize_images(tf.image.convert_image_dtype(image, dtype=tf.float32), [image_size, image_size]),
            label
        ))
    test_dataset = test_dataset.batch(test_images)
    test_iterator = test_dataset.make_initializable_iterator()
    test_image_batch, test_label_batch = test_iterator.get_next()

    # logit = inference(image_batch)
    # loss = calc_loss(logit, label_batch)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # initialize persistence class
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # print(sess.run(tf.cast(features['label'], tf.int32)))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("start training........")
        # for i in range(training_rounds):
        i = 0
        step = 0
        if doTrain:
            sess.run(iterator.initializer)
            sess.run(validation_iterator.initializer)
            while True:
                i += 1
                try:
                    # img = sess.run(distorted_img)
                    # plt.imshow(img)
                    # plt.show()

                    xs, ys = sess.run([image_batch, label_batch])
                    # print(xs.shape)
                    # print(ys.shape)
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

                    if i % 200 == 0:
                        vxs, vys = sess.run([validation_image_batch, validation_label_batch])
                        p, a, accuracy_score = sess.run([prediction, answer, accuracy], feed_dict={x: vxs, y_: vys})
                        print("prediction: \t%s, \nanswer: \t\t%s" % (p[0:10], a[0:10]))
                        print("after %d steps, loss: %.3f, accuracy: %.3f" % (step, loss_value, accuracy_score))
                except tf.errors.OutOfRangeError:
                    # i = step
                    break
            sess.run(test_iterator.initializer)
            tp = []
            ta = []
            while True:
                try:
                    txs, tys = sess.run([test_image_batch, test_label_batch])
                    p, a = sess.run([prediction, answer], feed_dict={x: txs, y_: tys})
                    tp.extend(p)
                    ta.extend(a)
                except tf.errors.OutOfRangeError:
                    break

            correct = [float(y == y_) for (y, y_) in zip(tp, ta)]
            accuracy_score = sum(correct) / len(correct)
            print("in total %d steps, total accuracy: %.3f" % (i, accuracy_score))
            try:
                os.mkdir("model/")
            except:
                print("directory already exist")
            saver.save(
                sess, os.path.join("model/", "model.ckpt"), global_step=global_step
            )

        else:

            ckpt = tf.train.get_checkpoint_state("model/")
            if ckpt and ckpt.model_checkpoint_path:
                sess.run(test_iterator.initializer)
                saver.restore(sess, ckpt.model_checkpoint_path)
                start = np.random.randint(int(test_images/3), int(test_images/2))
                length = 10
                txs, tys = sess.run([test_image_batch, test_label_batch])
                p, a = sess.run([prediction, answer], feed_dict={x: txs[start:start+length], y_: tys[start:start+length]})
                print("prediction: \t%s, \nanswer: \t\t%s" % (p, a))

            else:
                print("model not exist")
        coord.request_stop()
        coord.join(threads)


# ************* dataset operation **************
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'feat1': tf.FixedLenFeature([], tf.int64),
            'feat2': tf.FixedLenFeature([], tf.int64),
        })
    return features['feat1'], features['feat2']


def dataset_basic_test():
    # 从tensor构建数据集
    input_data = [1, 2, 3, 5, 8]
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    # traverse dataset
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    y = x * x

    # 从文本构建数据集
    # input_files = ["file1", "file2"]
    # dataset = tf.data.TextLineDataset(input_files)

    # 从tfrecord构建数据集
    input_files = ["file1", "file2"]
    dataset = tf.data.TFRecordDataset(input_files)
    # call parser and replace each element with returned value
    dataset = dataset.map(parser)
    # make_one_shot_iterator 所有参数必须确定, 使用placeholder需使用initializable_iterator
    # reinitializable_iterator, initialize multiple times for different data source
    # feedable_iterator, use feed_dict to assign iterators to run
    iterator = dataset.make_one_shot_iterator()
    feat1, feat2 = iterator.get_next()

    with tf.Session() as sess:
        # for i in range(len(input_data)):
        #     print(sess.run(y))

        for i in range(10):
            f1, f2 = sess.run([feat1, feat2])

    # 从tfrecord构建数据集, placeholder
    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)
    iterator = dataset.make_initializable_iterator()
    feat1, feat2 = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={
            input_files: ["file1", "file2"]
        })
        while True:
            try:
                sess.run([feat1, feat2])
            except tf.errors.OutOfRangeError:
                break

    # dataset high level API
    image_size = 299
    buffer_size = 1000  # min_after_dequeue
    batch_size = 100
    N = 10  # num_epoch
    dataset = dataset.map(
        lambda x: preprocessing.process_for_train(x, image_size, image_size, None)
    )
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(N)


if __name__ == '__main__':
    # threads_mgmt()
    # generate_files()
    # read_files()
    # batch_example()
    # process_data()
    # generate_record()
    process_data(doTrain=False)
    # dataset_basic_test()
