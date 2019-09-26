import tensorflow as tf
import numpy as np
import threading
import time


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


if __name__ == '__main__':
    threads_mgmt()
