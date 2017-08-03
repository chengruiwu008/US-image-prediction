import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import scipy.signal as signal
lenth=96
n_inputs=lenth*lenth*128//1024
n_hidden_units = 1024
n_outputs=n_inputs

def gray2binary(a):
    for i in range(len(a)):
        if a[i]>60:
            a[i]=1
        elif a[i]<=60:
            a[i]=0
    return a

def get_train_batch():
    ran = np.random.randint(0,8520,size=10,dtype='int')
    image_out = []
    label = []
    n_pic = ran # np.random.randint(0,5980)
    # print(n_pic)
    for i in range(10):
        image = []
        for j in range(4):
            frame_0 = cv2.imread('./lzc_friendship_20170730_050909_us/%d.jpg' % (n_pic[i] + j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0 / 255.0
            image.append(frame_0)
            # print('shape(image)',np.shape(image))
        image_out.append(image)
        # print('shape(image_out)',np.shape(image_out))
    for i in range(10):
        frame_1 = cv2.imread('./lzc_friendship_20170730_050909_us/%d.jpg' % (n_pic[i] + 4), 0)
        frame_1 = cv2.resize(frame_1, (lenth,lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        # frame_1 = gray2binary(frame_1)
        frame_1 = frame_1 / 255.0
        label.append(frame_1)
    return np.array(image_out,dtype='float') , np.array(label,dtype='float')

def get_test_batch(): # n_pic[600,5996]
    ran = np.random.randint(0, 9930, size=10, dtype='int')
    image_out = []
    label = []
    n_pic = ran  # np.random.randint(0,5980)
    # print(n_pic)
    for i in range(10):
        image = []
        for j in range(4):
            frame_0 = cv2.imread('./syf_friendship_20170731_153206_us/%d.jpg' % (n_pic[i] + j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0 / 255.0
            image.append(frame_0)
            # print('shape(image)', np.shape(image))
        image_out.append(image)
        # print('shape(image_out)', np.shape(image_out))
    for i in range(10):
        frame_1 = cv2.imread('./syf_friendship_20170731_153206_us/%d.jpg' % (n_pic[i] + 4), 0)
        frame_1 = cv2.resize(frame_1, (lenth, lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = frame_1 / 255.0
        # frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image_out, dtype='float'), np.array(label, dtype='float'), n_pic

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

inputs_ = tf.placeholder(tf.float32, [None, lenth,lenth, 1])
targets_ = tf.placeholder(tf.float32, [None, lenth,lenth, 1])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))}

conv1 = tf.layers.conv2d(inputs_, 32, (5,5), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
conv1 = batch_norm(conv1,32)
conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
conv2 = batch_norm(conv2,64)
conv3 = tf.layers.conv2d(conv2, 64, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
conv3 = batch_norm(conv3,64)
conv4 = tf.layers.conv2d(conv3, 128, (3,3), padding='same', activation=tf.nn.relu)
conv4 = tf.layers.max_pooling2d(conv4, (2,2), (2,2), padding='same')
conv4 = batch_norm(conv4,128)
conv5 = tf.layers.conv2d(conv4, 128, (3,3), padding='same', activation=tf.nn.relu)
conv5 = tf.layers.max_pooling2d(conv5, (2,2), (2,2), padding='same')
conv5 = batch_norm(conv5,128)
# print('conv5',conv5.shape)

conv5 = tf.reshape(conv5, [10,4,n_inputs])
cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(10, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, conv5, initial_state=init_state, time_major=False)
outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = tf.reshape(pred,[-1,lenth//32,lenth//32,128])
# print('pred',pred.shape)
conv6 = tf.image.resize_nearest_neighbor(pred, (6,6))
conv6 = tf.layers.conv2d(conv6, 128, (3,3), padding='same', activation=tf.nn.relu)
conv6 = batch_norm(conv6,128)
conv7 = tf.image.resize_nearest_neighbor(conv6, (12,12))
conv7 = tf.layers.conv2d(conv7, 64, (3,3), padding='same', activation=tf.nn.relu)
conv7 = batch_norm(conv7,64)
conv8 = tf.image.resize_nearest_neighbor(conv7, (24,24))
conv8 = tf.layers.conv2d(conv8, 64, (3,3), padding='same', activation=tf.nn.relu)
conv8 = batch_norm(conv8,64)
conv9 = tf.image.resize_nearest_neighbor(conv8, (48,48))
conv9 = tf.layers.conv2d(conv9, 32, (3,3), padding='same', activation=tf.nn.relu)
conv9 = batch_norm(conv9,32)
conv10 = tf.image.resize_nearest_neighbor(conv9, (96,96))
conv10 = tf.layers.conv2d(conv10, 1, (5,5), padding='same', activation=None)# tf.nn.relu)
# print('conv10',conv10.shape)
# outputs_ = tf.nn.softmax(logits_, dim= -1,name='outputs_')
# outputs_ = outputs_[:,:,:,0]
outputs_ = tf.reshape(conv10 , [-1,lenth,lenth,1])

cost = tf.reduce_mean(tf.square(tf.reshape(targets_,[-1]) - tf.reshape(outputs_,[-1])))
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=outputs_)
# cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

all_saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./CNN_RNN/data.chkp.meta')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    loss_history=[]
    saver.restore(sess, tf.train.latest_checkpoint('./CNN_RNN/'))
    for i in range(10000): #10000
        batch, img = get_train_batch()
        batch= np.reshape(batch,[-1, lenth,lenth, 1])
        img= np.reshape(img,[-1,lenth,lenth, 1])
        sess.run(optimizer, feed_dict={inputs_: batch, targets_: img})
        if i % 200 == 0:
            batch_cost = sess.run(cost, feed_dict={inputs_: batch, targets_: img})
            #loss_history.append(batch_cost)
            print("Batch: {} ".format(i), "Training loss: {:.4f}".format(batch_cost))
            all_saver.save(sess, './CNN_RNN/data.chkp')
    print("Optimization Finishes!")

    for i in range(10000): #[600,5996]
        batch_xs, batch_ys, n_pic = get_test_batch()
        batch_xs = np.reshape(batch_xs,[-1, lenth,lenth, 1])
        batch_ys = np.reshape(batch_ys, [-1, lenth,lenth, 1])
        image_p = sess.run(outputs_, feed_dict={inputs_: batch_xs, targets_: batch_ys})
        image_p = image_p * 255
        image_p = np.array(image_p ,dtype='int').reshape([-1,96,96])
        # print(np.shape(image_p))
        for n in range(10):
            img = np.array(np.reshape(image_p[n], (lenth,lenth)),dtype='int32')
            cv2.imwrite("./CNN_RNN/CNN_RNN_image_predict/%d.jpg" % (n_pic[n] + 4), img) #, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if i%100==0:
            print('%d finished!'% i)
        # image_p = gray2binary(image_p)
    # plt.figure(0)
    # f_1, b = plt.subplots(3, 11, figsize=(11, 3))
    # for i in range(11):
    #     # a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
    #     b[0][i].imshow(np.reshape(us_0[i], (lenth,lenth)))
    #     b[1][i].imshow(np.reshape(snake_0[i], (lenth,lenth)))
    #     b[2][10].imshow(np.reshape(image_p[0], (lenth,lenth)))
    #
    # plt.figure(1)
    # f, a = plt.subplots(3, 10, figsize=(10, 3))
    # for i in range(10):
    #     #a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
    #     a[0][i].imshow(np.reshape(label_0[i], (lenth,lenth)))
    #     a[1][i].imshow(np.reshape(batch_ys[i], (lenth,lenth)))
    #     a[2][i].imshow(np.reshape(image_p[i], (lenth,lenth)))
    # plt.show()