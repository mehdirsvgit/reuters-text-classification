import tensorflow as tf
y  = tf.constant([0, 1, 0])
x  = tf.constant([0, 1, 1])
all_elems_equal = tf.reduce_mean(tf.cast(tf.equal(y, x), "float"))

with tf.Session() as sess:
    b = sess.run(all_elems_equal)
    if b:
        print('true')
    else:
        print('false')