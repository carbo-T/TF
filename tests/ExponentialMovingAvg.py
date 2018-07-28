# improve the robustness of stochastic gradient descend training system using EMA func
import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)

# animate iterations in nn, control decay dynamically
step = tf.Variable(0, trainable=False)

# decay is 0.99 by default
ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_average_op = ema.apply([v1])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    # output [0,0]
    print sess.run([v1, ema.average(v1)])

    sess.run(tf.assign(v1, 5))
    sess.run(maintain_average_op)
    # decay = min(0.99, 1+step/10+step) = 0.1, var = 0*0.1+0.9*5 = 4.5
    # output [5,4.5]
    print sess.run([v1, ema.average(v1)])

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_average_op)
    # decay = min(0.99, 1+step/10+step) = 0.99, var = 4.5*0.99+10*0.01 = 4.555
    # output [10, 4.555]
    print sess.run([v1, ema.average(v1)])

    sess.run(maintain_average_op)
    # decay = min(0.99, 1+step/10+step) = 0.99, var = 4.555*0.99+10*0.01 = 4.60945
    # output [10, 4.60945]
    print sess.run([v1, ema.average(v1)])

