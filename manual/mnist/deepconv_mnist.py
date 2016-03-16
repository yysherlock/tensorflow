import tensorflow as tf
import input_data

# Set up
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mnist is a lightweight class which stores the training, validation, and testing sets as Numpy arrays.
    # It also provides a function for iterating through data minibatches: `next`, which we will use below.

# Tensorflow relies on a highly efficient C++ backend to do its computation. The connection to this backend is
# called a session. The common usage for TensorFlow programs is to first create a graph and then 
# launch it in a session.

# InteractiveSession class makes TensorFlow more flexible about how you structure your code. It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.

sess = tf.InteractiveSession()

# The role of Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run.

# Build a Softmax Regression Model
# First, we will build a softmax regression model with a single linear layer.
# Second, we will extend this to the case of softmax regression with a multilayer convolutional network.

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
print type(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
print type(train_step)

# Before Variables can be used within a session, they must be initialized using that session.
sess.run(tf.initialize_all_variables())

for i in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch = mnist.train.next_batch(100)
    #sess.run(train_step, feed_dict={x:batch[0], y_:batch[1]})
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# Evaluate on test data
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1)), "float"))
print type(accuracy)
#print sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
print accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
# Tensor.eval(feed_dict) or Operation.run(feed_dict)

# To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. 

# Weight Initialization

# One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial) # initial is a Tensor

# Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons".
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Convolution and Pooling
# TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input. Our pooling is plain old amx pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations into functions.


