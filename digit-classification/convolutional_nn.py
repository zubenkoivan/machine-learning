import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial_value = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_value)


def bias_variable(shape):
    initial_value = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial_value)


def print_accuracy(prefixText, accuracy):
    print(prefixText + " accuracy = {}%".format(round(accuracy * 100, 2)))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
dropout = tf.placeholder(tf.float32)

# convolutional layer 1
X_conv1 = tf.reshape(X, [-1, 28, 28, 1])
W_conv1 = weight_variable([6, 6, 1, 6])
b_conv1 = bias_variable([6])
Y_conv1 = tf.nn.relu(tf.nn.conv2d(X_conv1, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

# convolutional layer 2
W_conv2 = weight_variable([5, 5, 6, 12])
b_conv2 = bias_variable([12])
Y_conv2 = tf.nn.relu(tf.nn.conv2d(Y_conv1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)

# convolutional layer 3
W_conv3 = weight_variable([4, 4, 12, 24])
b_conv3 = bias_variable([24])
Y_conv3 = tf.nn.relu(tf.nn.conv2d(Y_conv2, W_conv3, strides=[1, 2, 2, 1], padding="SAME") + b_conv3)

# fully connected layer 1
X_full1 = tf.reshape(Y_conv3, [-1, 7 * 7 * 24])
W_full1 = weight_variable([7 * 7 * 24, 200])
b_full1 = bias_variable([200])
Y_full1 = tf.nn.relu(tf.matmul(X_full1, W_full1) + b_full1)
Y_full1 = tf.nn.dropout(Y_full1, dropout)

# fully connected layer 2
W_full2 = weight_variable([200, 10])
b_full2 = bias_variable([10])
Y_predicted_logits = tf.matmul(Y_full1, W_full2) + b_full2
Y_predicted = tf.nn.softmax(Y_predicted_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_predicted_logits, labels=Y)
mean_cross_entropy = tf.reduce_mean(cross_entropy)

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
end_learning_rate = 0.0001
decay_steps = 10000
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps, end_learning_rate, power=0.5)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_data = {X: mnist.test.images, Y: mnist.test.labels, dropout: 1.0}

    for i in range(10000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y: batch_Y, dropout: 0.5}

        if i % 100 == 0:
            iteration = int(i / 100 + 1)
            # print_accuracy(str(iteration) + ". Train", sess.run(accuracy, feed_dict={X: batch_X, Y: batch_Y, dropout: 1.0}))
            print_accuracy(str(iteration) + ". Test", sess.run(accuracy, feed_dict=test_data))
        else:
            sess.run(train_step, feed_dict=train_data)

    print_accuracy("Test", sess.run(accuracy, feed_dict=test_data))

