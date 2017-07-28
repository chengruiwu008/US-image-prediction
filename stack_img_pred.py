import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

lenth=96
n_inputs=lenth*lenth*32//64
n_hidden_units = 2048
n_outputs=n_inputs

def gray2binary(a):
    for i in range(len(a)):
        if a[i]>60:
            a[i]=1
        elif a[i]<=60:
            a[i]=0
    return a

def get_train_batch():
    ran = np.random.randint(600,6000,size=10,dtype='int')
    image = []
    label = []
    n_pic = ran #np.random.randint(600,5996)
    # print(n_pic)
    for i in range(10):
        image_0=[]
        for j in range(4):
            frame_0 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic[i]+j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0 / 255.0
            image_0.append(frame_0)
        image.append(image_0)
    # print(np.shape(image))
    # np.transpose(image,[1,0,2])
    # print(np.shape(image))
    for i in range(10):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic[i]+4), 0)
        frame_1 = cv2.resize(frame_1, (lenth,lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    return np.array(image,dtype='float') , np.array(label,dtype='float')

def get_test_batch(n_pic): # n_pic[600,5996]
    #ran = np.random.randint(5800,6000, size=10, dtype='int')
    image = []
    label = []
    label_0 = []
    #n_pic = ran  # np.random.randint(600,5996)
    for i in range(10):
        image_0 = []
        for j in range(4):
            frame_0 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic + i + j), 0)
            # frame_0 = add_noise(frame_0, n = noise)
            frame_0 = cv2.resize(frame_0, (lenth, lenth))
            frame_0 = np.array(frame_0).reshape(-1)
            frame_0 = frame_0 / 255.0
            image_0.append(frame_0)
        image.append(image_0)
    for i in range(10):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic + i + 4), 0)
        frame_1 = cv2.resize(frame_1, (lenth, lenth))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        label.append(frame_1)
    for i in range(10):
        frame_2 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic + i + 4), 0)
        frame_2 = cv2.resize(frame_2, (lenth, lenth))
        frame_2 = np.array(frame_2).reshape(-1)
        frame_2 = frame_2 / 255.0
        label_0.append(frame_2)
    return np.array(image, dtype='float'), np.array(label, dtype='float'), np.array(label_0, dtype='float')

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

inputs_ = tf.placeholder(tf.float32, [None, lenth,lenth, 4])
targets_ = tf.placeholder(tf.float32, [None, lenth,lenth, 1])

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

conv6 = tf.image.resize_nearest_neighbor(conv5, (6,6))
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
conv10 = tf.layers.conv2d(conv10, 2, (5,5), padding='same', activation=None)# tf.nn.relu)
#conv10 = batch_norm(conv10,2)
#logits_ = tf.layers.conv2d(conv6, 2, (3,3), padding='same', activation=None)
outputs_ = tf.nn.softmax(conv10, dim= -1,name='outputs_')
outputs_ = outputs_[:,:,:,0]
outputs_ = tf.reshape(outputs_ , [-1,lenth,lenth,1])

cost = tf.reduce_mean(tf.square(tf.reshape(targets_,[-1]) - tf.reshape(outputs_,[-1])))
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=outputs_)
# cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
all_saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./stack_img_pred_saver/data.chkp.meta')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, tf.train.latest_checkpoint('./stack_img_pred_saver/')) # './stack_img_pred_saver/data.chkp.data-00000-of-00001')
    loss_history=[]
    for i in range(10000): #10000
        batch, img = get_train_batch()
        #print(np.shape(batch))
        batch = np.transpose(batch,[0,2,1])
        batch= np.reshape(batch,[-1, lenth,lenth, 4])
        img= np.reshape(img,[-1,lenth,lenth, 1])
        sess.run(optimizer, feed_dict={inputs_: batch, targets_: img})
        if i % 100 == 0:
            batch_cost = sess.run(cost, feed_dict={inputs_: batch, targets_: img})
            loss_history.append(batch_cost)
            print("Batch: {} ".format(i), "Training loss: {:.4f}".format(batch_cost))
            all_saver.save(sess, './stack_img_pred_saver/data_0.chkp')
    print("Optimization Finishes!")

    for i in range(600,6000,10):
        batch_xs, batch_ys, batch_us = get_test_batch(i)
        batch_xs = np.transpose(batch_xs, [0, 2, 1])
        batch_xs = np.reshape(batch_xs, [-1, lenth, lenth, 4])
        batch_ys = np.reshape(batch_ys, [-1, lenth, lenth, 1])
        batch_us = np.reshape(batch_us, [-1, lenth, lenth, 1])
        image_p = sess.run(outputs_, feed_dict={inputs_: batch_xs, targets_: batch_ys})
        # image_p = gray2binary(image_p)
        # plt.figure(2)
        # plt.plot(loss_history)  # , '-o')
        # plt.xlabel('train')
        # plt.ylabel('loss')

        for j in range(10):
            plt.figure('pred_g', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(image_p[j], (lenth, lenth)), cmap='gray')
            plt.savefig("./image_predict_00/pred_gray%d.jpg" % (i+j))

        for j in range(10):
            plt.figure('snake_g', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(batch_ys[j], (lenth, lenth)), cmap='gray')
            plt.savefig("./image_predict_00/snake_gray%d.jpg" % (i+j))

        for j in range(10):
            plt.figure('pred', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(image_p[j], (lenth, lenth)))
            plt.savefig("./image_predict_00/pred%d.jpg" % (i+j))

        for j in range(10):
            plt.figure('snake', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(batch_ys[j], (lenth, lenth)))
            plt.savefig("./image_predict_00/snake%d.jpg" % (i+j))

        for j in range(10):
            plt.figure('us_g', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(batch_us[j], (lenth, lenth)), cmap='gray')
            plt.savefig("./image_predict_00/us_gray%d.jpg" % (i + j))

        for j in range(10):
            plt.figure('us', figsize=(10, 10))
            plt.axis('off')
            plt.imshow(np.reshape(batch_us[j], (lenth, lenth)))
            plt.savefig("./image_predict_00/us%d.jpg" % (i + j))

        print('%d of 5400 finished' % (i-599))
        # plt.figure(1)
        # f, a = plt.subplots(2, 10, figsize=(10, 2))
        # for i in range(10):
        #     # a[0][i].imshow(np.reshape(ys_0[i], (LONGITUDE, LONGITUDE)))
        #     a[0][i].imshow(np.reshape(batch_ys[i], (lenth,lenth)))
        #     #a[1][i].imshow(np.reshape(batch_xs[i], (24, 24)))
        #     a[1][i].imshow(np.reshape(image_p[i], (lenth,lenth)))
        #     #a[1][0].save("./image_predict_00/%d.jpg" % 0)
        # plt.show()
    print('All finished!')
