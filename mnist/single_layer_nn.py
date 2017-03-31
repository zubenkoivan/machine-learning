import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, 10])

W = tf.Variable(tf.zeros([784, 10], dtype=tf.float64))
b = tf.Variable(tf.zeros([10], dtype=tf.float64))

Y_predicted = tf.nn.softmax(tf.matmul(X, W) + b)
# cross_entropy = tf.reduce_sum(-Y * tf.log(Y_predicted), reduction_indices=[1])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(X, W) + b, labels=Y)
mean_cross_entropy = tf.reduce_mean(cross_entropy)

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float64))

train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

with tf.InteractiveSession() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y: batch_Y}
        sess.run(train_step, feed_dict=train_data)

    a = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Test accuracy = {}%".format(a * 100))
