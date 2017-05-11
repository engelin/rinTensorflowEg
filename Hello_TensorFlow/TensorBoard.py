# Use the TensorBoard

import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# with tf.name_scope
with tf.name_scope('layer1'):
   W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name="W1")
   L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
   W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name="W2")
   L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
   W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name="W3")
   model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
   optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
   train_op = optimizer.minimize(cost)

   # tf.summary.scalar
   tf.summary.scalar('cost', cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# TensorBoard
# tf.summary.merge_all
merged = tf.summary.merge_all()
# tf.summary.FileWriter
writer = tf.summary.FileWriter('./logs', sess.graph)
# Verify logs
# $ tensorboard --logdir=./logs
# http://localhost:6006


for step in xrange(100):
   sess.run(train_op, feed_dict={X: x_data, Y: y_data})
   # merge and store values
   summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
   writer.add_summary(summary, step)


prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print "prediction value: ", sess.run(prediction, feed_dict={X: x_data})
print "target value: ", sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

print "accuracy: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})
