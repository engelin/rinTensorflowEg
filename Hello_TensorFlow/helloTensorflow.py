# Exercise Tensorflow

import tensorflow as tf

# tf.constant
hello = tf.constant('Hello, TensorFlow!')

const1 = tf.constant(10)
const2 = tf.constant(20)
result = const1 + const2


# tf.placeholder
X = tf.placeholder("float", [None, 3])

# tf.Variable
# tf.random_normal
W = tf.Variable(tf.random_normal([3, 2]), name='Weights')
b = tf.Variable(tf.random_normal([2, 1]), name='Bias')

x_data = [[1, 2, 3], [4, 5, 6]]

# tf.matmul
expr = tf.matmul(X, W) + b

# tf.Session
sess = tf.Session()
# sess.run
# tf.global_variables_initializer
sess.run(tf.global_variables_initializer())

print "Constants: "
print sess.run(hello)
print "const1 + const2 = ", sess.run(result)

print "x_data: ", x_data
print "W: ", sess.run(W)
print "b: ", sess.run(b)
print "expr: ", sess.run(expr, feed_dict={X: x_data})
# 'expr' needs input variable X
# Running expr, use like this

sess.close()
