import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.matmul(input1,input2)

with tf.Session() as sess:
    # 用placehoolder则在run的时候输入值，所以是和feed_dict绑定的
    print(sess.run(output,feed_dict={input1:[7],input2:[3.]}))