import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial_value = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_value)


def bias_variable(shape):
    initial_value = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial_value)


mnist = input_data.read_data_sets("MNIST_data_set/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout = tf.placeholder(tf.float32)

N1 = 200
W1 = weight_variable([784, N1])
b1 = bias_variable([N1])
Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Y1 = tf.nn.dropout(Y1, dropout)

N2 = 100
W2 = weight_variable([N1, N2])
b2 = bias_variable([N2])
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y2 = tf.nn.dropout(Y2, dropout)

N3 = 60
W3 = weight_variable([N2, N3])
b3 = bias_variable([N3])
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y3 = tf.nn.dropout(Y3, dropout)

N4 = 30
W4 = weight_variable([N3, N4])
b4 = bias_variable([N4])
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Y4 = tf.nn.dropout(Y4, dropout)

W5 = weight_variable([N4, 10])
b5 = bias_variable([10])
Y_predicted_logits = tf.matmul(Y4, W5) + b5
Y_predicted = tf.nn.softmax(Y_predicted_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_predicted_logits, labels=Y)
mean_cross_entropy = tf.reduce_mean(cross_entropy)

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y: batch_Y, dropout: 0.75}
        sess.run(train_step, feed_dict=train_data)

    a = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, dropout: 1.0})
    print("Test accuracy = {}%".format(a * 100))
