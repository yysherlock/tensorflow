import tensorflow as tf
# Variables maintain state across executions of the graph.
# This example shows a variable serving as a simple counter.

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name = "counter")
print('state: ',type(state))

# Create an op to add one to `state`
one = tf.constant(1)
print('one: ',type(one))

new_value = tf.add(state, one)
print('new_value: ',type(new_value))
update = tf.assign(state, new_value) # assign new value to the variable
print('update: ',type(update))

# Variables must be initialized by running an `init` op after having launched the graph. We first have to add the `init` op to the graph.

init_op = tf.initialize_all_variables()

# lanuch the graph and run the ops
with tf.Session() as sess:
    sess.run(init_op) # run the `init` op
    print(sess.run(state)) # print the initial value of `state`

    # Run the op that updates `state` and print `state`
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        print(sess.run(update))
        print(type(sess.run(state)))
        print(type(sess.run(update)))

