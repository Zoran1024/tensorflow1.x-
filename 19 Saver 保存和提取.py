import tensorflow as tf

# Save to file
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_psth = saver.save(sess,'mynet/save_net.ckpt')
#     print("Save to path",save_psth)


# restore variable
import numpy as np
w = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'mynet/save_net.ckpt')
    print("weights:",sess.run(w))
    print('biases',sess.run(b))