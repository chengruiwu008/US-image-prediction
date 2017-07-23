import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

LONGITUDE=24
WIDTH=24
PIXEL=LONGITUDE*WIDTH
BATCHSIZE=10
CONV_FRAME=1
TIMESTEP=10
frame_size = 3

n_inputs = PIXEL
n_hidden_units = 2048
n_outputs = PIXEL
n_steps = 10
batch_size = 10
lr = 0.001
training_iters = 10000
data_size = 6000

def gray2binary(a):
    for i in range(len(a)):
        if a[i]>127:
            a[i]=1
        elif a[i]<=127:
            a[i]=0
    return a

def get_batch():
    ran = random.randint(1, data_size)
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(batch_size * n_steps):
        frame_0 = cv2.imread('./hardPixelImage/%d.jpg' % (n_pic+i), 0)
        frame_0 = cv2.resize(frame_0, (24, 24))
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(batch_size):
        frame_1 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_1 = cv2.resize(frame_1, (24, 24))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    for i in range(batch_size):
        frame_2 = cv2.imread('./hardPixelImage/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_2 = cv2.resize(frame_2, (24, 24))
        frame_2 = np.array(frame_2).reshape(-1)
        label_0.append(frame_2)
    return image , label , label_0

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

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
X_in = tf.reshape(x, [-1, n_inputs])
# into hidden
# X_in ==> ( batchsize * 10 steps, 1024 hidden)
X_in = tf.matmul(X_in, weights['in']) + biases['in']
# X_in ==> ( batchsize , 10 steps, 1024 hidden)
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
pred_0 = tf.matmul(outputs[-1], weights['out']) + biases['out']
outputs = tf.reshape(pred_0,[-1,24,24,1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

W_conv=weight_variable([frame_size,frame_size,1,2])
b_conv=bias_variable([2])
x_image=tf.reshape(outputs,[-1,LONGITUDE,WIDTH,1])
h_conv= tf.nn.relu(conv2d(x_image, W_conv) + b_conv) #shape=[-1,24,24,2]

pred = tf.nn.softmax(tf.reshape(h_conv,[-1,2]))
pred = tf.unstack(pred,axis=1)
pred = tf.reshape(pred[0],[-1,24,24,1])
pred_255 = pred * 255
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=tf.reshape(pred,[-1,n_outputs]), labels=y))

#pred = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (batchsize, 576)
# pred = tf.nn.softmax(pred)
# cost = tf.reduce_mean(tf.pow(pred - y, 2))

train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#loss_history=[]


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys ,_ = get_batch()
        batch_xs = np.reshape(batch_xs,[batch_size, n_steps, n_inputs])
        sess.run(train_op , feed_dict = { x: batch_xs ,y: batch_ys })
        # plt.ion()
        # plt.show()
        if step % 20 == 0:
            loss = sess.run(cross_entropy, feed_dict={x: batch_xs ,y: batch_ys })
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

    batch_xs, batch_ys ,ys_0= get_batch()
    batch_xs = np.reshape(batch_xs, [batch_size, n_steps, n_inputs])
    #batch_ys = np.reshape(batch_ys, [batch_size, n_steps, n_inputs])
    image_p = sess.run(pred_255, feed_dict={x: batch_xs, y: batch_ys })
    batch_ys = batch_ys * 255
    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(10):
        a[0][i].imshow(np.reshape(ys_0[i], (24, 24)))
        a[1][i].imshow(np.reshape(batch_ys[i], (24, 24)))
        a[2][i].imshow(np.reshape(image_p[i], (24, 24)))
        #a[3][i].imshow(np.reshape(image_p[i], (24, 24)))
    plt.show()
