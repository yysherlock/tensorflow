# Consider IPython, use `InteractiveSession` class to replace `Session` class.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
print type(x)

x.initializer.run()
print type(x)

sub = tf.sub(x, a)
print(sub.eval())

sess.close()
