import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

LONGITUDE=24
WIDTH=24
PIXEL=LONGITUDE*WIDTH
BATCHSIZE=10
CONV_FRAME=3
TIMESTEP=10

n_inputs = 6*6*32
n_hidden_units = 1024
n_outputs = PIXEL
n_steps = 10
batch_size = 10
lr = 0.001
training_iters = 20000
data_size = 5990

def get_batch(noise=1000):
    ran = random.randint(600, data_size)
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(batch_size * n_steps):
        frame_0 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic+i), 0)
        #frame_0 = add_noise(frame_0, n = noise)
        frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(batch_size):
        frame_1 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_1 = cv2.resize(frame_1, (LONGITUDE, LONGITUDE))
        frame_1 = np.array(frame_1).reshape(-1)
        label.append(frame_1)
    for i in range(batch_size):
        frame_2 = cv2.imread('./hardPixelImage/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_2 = cv2.resize(frame_2, (LONGITUDE, LONGITUDE))
        frame_2 = np.array(frame_2).reshape(-1)
        label_0.append(frame_2)
    return image , label , label_0

x = tf.placeholder(tf.float32, [n_steps, None, 24*24])
y = tf.placeholder(tf.float32, [None, n_outputs])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#CNN start

W_conv_1=[]
b_conv_1=[]
conv_1=[]
pool_1=[]
W_conv_2=[]
b_conv_2=[]
conv_2=[]
pool_2=[]
cnn_out=[]
x_image=[]
rnn_in=[]
for i in range(10):
    W_conv_1 = weight_variable([CONV_FRAME, CONV_FRAME, 1, 8])
    b_conv_1 = bias_variable([8])
    x_image = tf.reshape(x[i], [-1, LONGITUDE, WIDTH, 1])
    conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)
    pool_1 = max_pool_2x2(conv_1)

    W_conv_2 = weight_variable([CONV_FRAME, CONV_FRAME, 8, 32])
    b_conv_2 = bias_variable([32])
    conv_2 = tf.nn.relu(conv2d(pool_1, W_conv_2) + b_conv_2)
    pool_2 = max_pool_2x2(conv_2)
    cnn_out = tf.reshape(pool_2, [-1, 6*6*32])
    rnn_in.append(cnn_out)
rnn_in = tf.transpose(rnn_in, [1, 0, 2])

# RNN start
weights = {
    # (576, 1024)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (1024, 576)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))
}
biases = {
    # (1024, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (576, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}
# transpose the inputs shape from
# X ==> ( batchsize * 10 steps, 576 inputs)
X_in = tf.reshape(rnn_in, [-1, n_inputs])
# into hidden
# X_in ==> ( batchsize * 10 steps, 1024 hidden)
X_in = tf.matmul(X_in, weights['in']) + biases['in']
# X_in ==> ( batchsize , 10 steps, 1024 hidden)
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (1024, 576)
cost = tf.reduce_mean(tf.square(tf.reshape(pred,[-1]) - tf.reshape(y,[-1])))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys ,_ = get_batch(0)
        batch_xs = np.reshape(batch_xs,[batch_size, n_steps, 24*24])
        sess.run(train_op , feed_dict = { x: batch_xs ,y: batch_ys })
        # plt.ion()
        # plt.show()
        if step % 20 == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs ,y: batch_ys })
            print('step ',step,' cost = ',loss )
            #loss_history.append(loss)
            # plt.plot(loss_history, '-b')
            # plt.draw()
            # plt.pause(0.01)
        # image_p = sess.run(pred[-1], feed_dict={x: batch_xs ,y: batch_ys,})
        # image_p = tf.reshape(image_p, [LONGITUDE, WIDTH])
        # image_p = np.array( image_p , dtype = int)
        # cv2.imwrite('image_predict/' + str(step) + '.jpg', image_p)
        step += 1
    print("Optimization Finishes!")

    batch_xs, batch_ys, ys_0 = get_batch(0)
    batch_xs = np.reshape(batch_xs, [batch_size, n_steps, 24*24])
    # batch_ys = np.reshape(batch_ys, [batch_size, n_steps, n_inputs])
    image_p = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})
    # image_p = gray2binary(image_p)

    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(10):
        a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
        a[1][i].imshow(np.reshape(batch_ys[i], (LONGITUDE, LONGITUDE)))
        a[2][i].imshow(np.reshape(image_p[i], (LONGITUDE, LONGITUDE)))
        #a[3][i].imshow(np.reshape(gray2binary(image_p[i]), (LONGITUDE, LONGITUDE)))
    plt.show()

