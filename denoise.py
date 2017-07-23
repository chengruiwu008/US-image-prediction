import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
print("TensorFlow Version: %s" % tf.__version__)

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
data_size = 5800

def add_noise(img,n=1000):
    rows, cols = np.shape(img)
    for i in range(n):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y] = 255
    return np.array(img)

def get_train_batch(noise=0):
    ran = random.randint(600, data_size)
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(batch_size ):
        frame_0 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic+i), 0)
        frame_0 = add_noise(frame_0, n = noise)
        frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(batch_size):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_1 = cv2.resize(frame_1, (LONGITUDE, LONGITUDE))
        frame_1 = np.array(frame_1).reshape(-1)
        label.append(frame_1)
    return image , label

def get_test_batch():
    ran = random.randint(5800, 6000)
    # print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(10):
        frame_0 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic + i), 0)
        #frame_0 = add_noise(frame_0, n=noise)
        frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        # print(np.shape(image))
    for i in range(10):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic + batch_size * (i + 1)), 0)
        frame_1 = cv2.resize(frame_1, (LONGITUDE, LONGITUDE))
        frame_1 = np.array(frame_1).reshape(-1)
        label.append(frame_1)
    for i in range(10):
        frame_2 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_2 = cv2.resize(frame_2, (LONGITUDE, LONGITUDE))
        frame_2 = np.array(frame_2).reshape(-1)
        label_0.append(frame_2)
    return image, label, label_0

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')

conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
#conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

#conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
conv4 = tf.layers.conv2d(conv3, 32, (3,3), padding='same', activation=tf.nn.relu)

conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
# cost = tf.reduce_mean(loss)
cost = tf.reduce_mean(tf.square(tf.reshape(targets_,[-1]) - tf.reshape(logits_,[-1])))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
sess = tf.Session()

noise_factor = 0.5
epochs = 10

sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(100): #mnist.train.num_examples//batch_size
        batchs , imgs = get_train_batch()
        batch = tf.reshape(batchs, (-1, 28, 28, 1))
        img = tf.reshape(imgs, (-1, 28, 28, 1))
        # 加入噪声
        #noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(imgs, 0., 255.)
        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: batch,
                                            targets_: img})

        print("Epoch/Num: {}/{} ".format(e + 1, idx),
              "Training loss: {:.4f}".format(batch_cost))

fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20,4))
in_imgs, label, o_pic = get_test_batch()
#noisy_imgs = in_imgs + noise_factor * np.random.randn(*np.shape(in_imgs))
noisy_imgs = np.clip(in_imgs, 0., 1.)
#np.set_printoptions(threshold=10000000)
#print(noisy_imgs)

reconstructed = sess.run(outputs_,
                         feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})
#print(reconstructed)
for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

#fig.tight_layout(pad=0.1)
plt.show()
sess.close()