import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Type of mnist.train: ', type(mnist.train))
print(mnist.train)
print(mnist.test)

x = tf.placeholder("float", [None, 784]) # x is not a specific value, it's a placeholder, a value that we'll input
                        # when we ask TensorFlow to run a computation. None means that a dimension can be of any length.
y_ = tf.placeholder("float", [None, 10]) # correct labels

# We also need the weights and biases for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle it: Variable. A Variable is a modifiable tensor that lives in TensorFlow;s graph of interacting operations. It can be used and even modified by the computation. For machine learning applications, one generally has the model parameters be Variables.

W = tf.Variable(tf.zeros([784, 10])) # giving initial value to create tf Variable.
b = tf.Variable(tf.zeros([10]))

# Softmax Regression Model
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # tf.reduce_sum adds all the elements of the tensor

# TensorFlow knows the entire graph of your computations, it can automatically use the backpropagation algorithm to efficiently determine how your variables affect the cost you ask it minimize. Then it can apply your choice of optimization algorithm to modify the variables and reduce the cost.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # In this case, ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.01. Actually, add an operation to do gradient descent.

# Add an operation to initialize the variables we created
init = tf.initialize_all_variables()

# Now we launch the model in a Session, and  run the operation that initializes the variables.
sess = tf.Session()
sess.run(init)
# Let's train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict ={x : batch_xs, y_ : batch_ys})

# Evaluate our model on test data
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict = {x : mnist.test.images, y_ : mnist.test.labels})


