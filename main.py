from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
keep_prob = tf.placeholder(tf.float32)

y = tf.nn.softmax(tf.matmul(x, W) + b)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')


	
# first layer
conv1_W = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1))
conv1_b = tf.Variable(tf.constant(0.1, shape = [32]))
x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv2d(x_image, conv1_W) + conv1_b

activated1 = tf.nn.relu(conv1)
pooled1 = tf.nn.max_pool(activated1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# second layer
conv2_W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1))
conv2_b = tf.Variable(tf.constant(0.1, shape = [64]))
conv2 = conv2d(pooled1, conv2_W) + conv2_b

activated2 = tf.nn.relu(conv2)
pooled2 = tf.nn.max_pool(activated2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# connect
fc1_W = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev = 0.1))
fc1_b = tf.Variable(tf.constant(0.1, shape = [1024]))

fc1 = tf.reshape(pooled2, [-1, 7*7*64])
fc1_a = tf.nn.relu(tf.matmul(fc1, fc1_W) + fc1_b)

# dropout
fc1_a_drop = tf.nn.dropout(fc1_a, keep_prob)

# connect 2
fc2_W = tf.Variable(tf.truncated_normal([1024, 10]))
fc2_b = tf.Variable(tf.constant(0.1, shape = [10]))

conv_y = tf.matmul(fc1_a_drop, fc2_W) + fc2_b


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = conv_y, labels = y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(conv_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_x, batch_y = mnist.train.next_batch(50)
	if i % 50 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.0})
		print("step %d, training accuracy %g" % (i, train_accuracy))
	train_step.run(feed_dict = {x: batch_x, y_: batch_y, keep_prob: 0.5})
	
print("test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))