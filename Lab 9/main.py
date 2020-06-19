import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

# dataset size
n_train = mnist.train.num_examples 
n_validation = mnist.validation.num_examples 
n_test = mnist.test.num_examples

# network architecture
n_input = 784 
n_hidden1 = 512 
n_hidden2 = 256 
n_hidden3 = 128 
n_output = 10

# network parameters
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

# tf placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) # dropout

# w & b parameters
weights = {
	'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
	'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
	'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
	'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = {
	'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
	'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
	'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
	'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# network layers
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# define loss and optimiser
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define evaluation
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialise variables, start session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on minibatches
for i in range(n_iterations):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})
	
	# print loss and accuracy (per minibatch)
	if i%100==0:
		minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
		print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))


# accuracy on test set
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)

#inspired from: digitalocean.com and towardsdatascience.com