import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# np.linspace生成一个-1到1的300个数的等差数列，np.newaxis是增加一个维度由（300）-》（300，1）
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[300,1])
ys = tf.placeholder(tf.float32,[300,1])

l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

flg = plt.figure()
# 在一个画布上画多张图arg1: 在垂直方向同时画几张图  arg2: 在水平方向同时画几张图  arg3: 当前命令修改的是第几张图
ax = flg.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() # 让图片连续
plt.show() #show显示一次后程序就会暂停

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])  # 去除lines的第一条线段
        except:
            pass
        predition_value = sess.run(predition,feed_dict={xs:x_data,ys:y_data})
        lines = ax.plot(x_data,predition_value,'r-',lw=5)
        plt.pause(0.1) #暂停0.1秒

