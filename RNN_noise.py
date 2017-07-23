import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
import matplotlib.cm as cm
import scipy.signal as signal

LONGITUDE=24
WIDTH=24
PIXEL=LONGITUDE*WIDTH
BATCHSIZE=10
CONV_FRAME=3
TIMESTEP=10

n_inputs = PIXEL
n_hidden_units = 2048
n_outputs = PIXEL
n_steps = 10
batch_size = 10
lr = 0.001
training_iters = 20000
data_size = 5990

def add_noise(img,n=1000):
    rows, cols = np.shape(img)
    for i in range(n):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y] = 255
    return np.array(img)

def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

def add_gauss_noise(image,r=10):
    suanzi = np.fromfunction(func, (r, r), sigma=5)
    image = np.array(image)
    image2 = signal.convolve2d(image, suanzi, mode="same")
    image2 = (image2 / float(image2.max())) * 255
    return np.array(image2)

def get_batch(noise=10, r = 3):
    ran = random.randint(600, data_size)
    #print(ran)
    image = []
    label = []
    label_0 = []
    image_noise = []
    n_pic = ran
    # print(n_pic)
    for i in range(batch_size * n_steps):
        frame_0 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic+i), 0)
        frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        #frame_0 = np.array(frame_0).reshape(-1)
        #image_norm.append(frame_0)
        #frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        frame_0 = add_noise(frame_0, n=noise)
        frame_0 = add_gauss_noise(frame_0, r = r)
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(batch_size):
        frame_1 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_1 = cv2.resize(frame_1, (LONGITUDE, LONGITUDE))
        frame_1 = np.array(frame_1).reshape(-1)
        label.append(frame_1)
    for i in range(batch_size):
        frame_2 = cv2.imread('hardPixelImage/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_2 = cv2.resize(frame_2, (LONGITUDE, LONGITUDE))
        frame_2 = np.array(frame_2).reshape(-1)
        label_0.append(frame_2)
    for i in range(batch_size):
        frame_3 = cv2.imread('./easyPixelImage2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_3 = cv2.resize(frame_3, (LONGITUDE, LONGITUDE))
        frame_3 = add_noise(frame_3, n=noise)
        frame_3 = add_gauss_noise(frame_3, r=r)
        frame_3 = np.array(frame_3).reshape(-1)
        image_noise.append(frame_3)
    return image , label , label_0 , image_noise

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
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (1024, 576)
cost = tf.reduce_mean(tf.square(tf.reshape(pred,[-1]) - tf.reshape(y,[-1])))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

loss_history=[]

def gray2binary(a):
    for i in range(len(a)):
        if a[i]>60:
            a[i]=255
        elif a[i]<=60:
            a[i]=0
    return a

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys ,_ , _ = get_batch()
        batch_xs = np.reshape(batch_xs,[batch_size, n_steps, n_inputs])
        sess.run(train_op , feed_dict = { x: batch_xs ,y: batch_ys })
        # plt.ion()
        # plt.show()
        if step % 20 == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs ,y: batch_ys })
            print('step ',step,' cost = ',loss )
            loss_history.append(loss)
            # plt.plot(loss_history, '-b')
            # plt.draw()
            # plt.pause(0.01)
        # image_p = sess.run(pred[-1], feed_dict={x: batch_xs ,y: batch_ys,})
        # image_p = tf.reshape(image_p, [LONGITUDE, WIDTH])
        # image_p = np.array( image_p , dtype = int)
        # cv2.imwrite('image_predict/' + str(step) + '.jpg', image_p)
        step += 1
    print("Optimization Finishes!")

    batch_xs, batch_ys , ys_0, img_noise = get_batch()
    batch_xs = np.reshape(batch_xs, [batch_size, n_steps, n_inputs])
    #batch_ys = np.reshape(batch_ys, [batch_size, n_steps, n_inputs])
    image_p = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys })
    #image_p = gray2binary(image_p)

    f, a = plt.subplots(5, 10, figsize=(10, 5))
    for i in range(10):
        a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
        a[1][i].imshow(np.reshape(batch_ys[i], (LONGITUDE, LONGITUDE)))
        a[2][i].imshow(np.reshape(img_noise[i], (LONGITUDE, LONGITUDE)))
        a[3][i].imshow(np.reshape(image_p[i], (LONGITUDE, LONGITUDE)))
        a[4][i].imshow(np.reshape(gray2binary(image_p[i]), (LONGITUDE, LONGITUDE)))
    plt.show()