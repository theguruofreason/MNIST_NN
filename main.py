from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

logits = tf.matmul(x, W) + b
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
	batch_x, batch_y = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y})
	
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))