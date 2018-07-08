# tensor described with ( name shape type )
# data types:
#   int: tf.int8, 16, 32, 64, uint 8
#   float: tf.float32, float64
#   bool: tf.bool
#   complex: tf.complex64, complex128
import tensorflow as tf

a = tf.constant([1.0, 3.0], name="a")
b = tf.constant([3.0, 6.0], name="b")
sum = a+b
print tf.add(a, b, name="add")
with tf.Session().as_default():
    print (sum.eval())

with tf.Session() as sess:
    result = sess.run(a+b)
    print result

writer = tf.summary.FileWriter("logs", tf.get_default_graph())
writer.close()