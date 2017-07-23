import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import scipy.signal as signal

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

def gray2binary(a):
    for i in range(len(a)):
        if a[i]>60:
            a[i]=1
        elif a[i]<=60:
            a[i]=0
    return a

def get_train_batch(noise=500):
    ran = np.random.randint(600,5800,size=10,dtype='int')
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(10):
        frame_0 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic[i]), 0)
        #frame_0 = add_noise(frame_0, n = noise)
        frame_0 = cv2.resize(frame_0, (24, 24))
        frame_0 = np.array(frame_0).reshape(-1)
        frame_0 = frame_0 / 255.0
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(10):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic[i]), 0)
        frame_1 = cv2.resize(frame_1, (24, 24))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image,dtype='float') , np.array(label,dtype='float')

def get_test_batch(noise=500):
    ran = np.random.randint(5800,6000,size=10,dtype='int')
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(10):
        frame_0 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic[i]), 0)
        #frame_0 = add_noise(frame_0, n = noise)
        frame_0 = cv2.resize(frame_0, (24, 24))
        frame_0 = np.array(frame_0).reshape(-1)
        frame_0 = frame_0 / 255.0
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(10):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic[i]), 0)
        frame_1 = cv2.resize(frame_1, (24, 24))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image,dtype='float') , np.array(label,dtype='float')

def input_norm(xs):
    fc_mean, fc_var = tf.nn.moments(
        xs,
        axes=[0],
    )
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    # apply moving average for mean and var when train on batch
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = mean_var_with_update()
    xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return xs

def batch_norm(Wx_plus_b,out_size):
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_b,
        axes=[0],  # the dimension you wanna normalize, here [0] for batch
        # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
    )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    # apply moving average for mean and var when train on batch
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = mean_var_with_update()
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
    return Wx_plus_b

inputs_ = tf.placeholder(tf.float32, [None, 24, 24, 1])
targets_ = tf.placeholder(tf.float32, [None, 24, 24, 1])

#inputs_ = input_norm(inputs_)

conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
conv1 = batch_norm(conv1,64)
conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
conv2 = batch_norm(conv2,64)
conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
conv3 = batch_norm(conv3,32)
conv4 = tf.image.resize_nearest_neighbor(conv3, (6,6))
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)
conv4 = batch_norm(conv4,32)
conv5 = tf.image.resize_nearest_neighbor(conv4, (12,12))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)
conv5 = batch_norm(conv5,64)
conv6 = tf.image.resize_nearest_neighbor(conv5, (24,24))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)
conv6 = batch_norm(conv6,64)
logits_ = tf.layers.conv2d(conv6, 2, (3,3), padding='same', activation=None)
outputs_ = tf.nn.softmax(logits_, dim= -1,name='outputs_')
outputs_ = outputs_[:,:,:,0]
outputs_ = tf.reshape(outputs_ , [-1,24,24,1])

cost = tf.reduce_mean(tf.square(tf.reshape(targets_,[-1]) - tf.reshape(outputs_,[-1])))
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=outputs_)
# cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)
#sess = tf.Session()

#sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        batch, img = get_train_batch()
        batch= np.reshape(batch,[-1, 24, 24, 1])
        img= np.reshape(img,[-1, 24, 24, 1])
        sess.run(optimizer, feed_dict={inputs_: batch, targets_: img})
        if i % 20 == 0:
            batch_cost = sess.run(cost, feed_dict={inputs_: batch, targets_: img})
            print("Batch: {} ".format(i), "Training loss: {:.4f}".format(batch_cost))
    print("Optimization Finishes!")

    batch_xs, batch_ys = get_test_batch()
    batch_xs = np.reshape(batch_xs,[-1, 24, 24, 1])
    batch_ys = np.reshape(batch_ys, [-1, 24, 24, 1])
    image_p = sess.run(outputs_, feed_dict={inputs_: batch_xs, targets_: batch_ys})
    # image_p = gray2binary(image_p)

    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(10):
        #a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
        a[0][i].imshow(np.reshape(batch_ys[i], (24, 24)))
        a[1][i].imshow(np.reshape(batch_xs[i], (24, 24)))
        a[2][i].imshow(np.reshape(image_p[i], (24, 24)))

        # a[3][i].imshow(np.reshape(gray2binary(image_p[i]), (LONGITUDE, LONGITUDE)))
    plt.show()

