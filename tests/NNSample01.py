import tensorflow as tf
from numpy.random import RandomState

# define the size of a batch
batch_size=4

# define coefficient matrices
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# define place for input and output, use param 'None' in shape can make the placeholder more flexible
x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

# forward propagation
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# define loss function ( sigmoid : 1/1+exp(-x) ), cross_entropy and train_step
y = tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# create a simulated dataset with a random number generator
rdm = RandomState(1)
dataset_size = 1280
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2<1)] for (x1, x2) in X]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print w1.eval(session=sess)
    print sess.run(w2)

    # writer = tf.summary.FileWriter("logs", tf.get_default_graph())
    # set the number of iteration
    STEPS = 50000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+ batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i%1000==0:
            # calculate cross entropy with some interval
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print ("after %d training step(s), cross entropy on all data is %g." % (i, total_cross_entropy))
            # tf.summary.histogram("iteration-w1", w1)
            # tf.summary.histogram("iteration-w2", w2)

    print sess.run(w1)
    print sess.run(w2)


# writer.close()
