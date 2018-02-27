import tensorflow as tf

a = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]], tf.float64)
b = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]], tf.float64)

loss = tf.losses.sigmoid_cross_entropy(a, b,)

s = tf.Session()

print(s.run(loss))

