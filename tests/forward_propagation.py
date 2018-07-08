import tensorflow as tf

# define two variables w1 and w2 as weight matrices, use seed to guarantee we get constant result.

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# define input eigenvector as a constant vector
# x = tf.constant([[0.7, 0.9]])

# use placeholder to store data in a constant place rather than create a large number of variables
x = tf.placeholder(tf.float32,shape=[3, 2], name="input")

# forward propagation to receive the output
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    sess.run(tf.global_variables_initializer())
    print (sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
