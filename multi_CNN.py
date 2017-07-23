import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

LONGITUDE=96
WIDTH=96
PIXEL=LONGITUDE*WIDTH
BATCHSIZE=10
CONV_FRAME=3
TIMESTEP=10

n_inputs = PIXEL
n_hidden_units = 3000
n_outputs = PIXEL
n_steps = 10
batch_size = 10
lr = 0.001
training_iters = 10000
data_size = 5990

def get_batch():
    ran = random.randint(600, data_size)
    #print(ran)
    image = []
    label = []
    label_0 = []
    n_pic = ran
    # print(n_pic)
    for i in range(batch_size * n_steps):
        frame_0 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic+i), 0)
        frame_0 = cv2.resize(frame_0, (LONGITUDE, LONGITUDE))
        frame_0 = np.array(frame_0).reshape(-1)
        image.append(frame_0)
        #print(np.shape(image))
    for i in range(batch_size):
        frame_1 = cv2.imread('./cropedoriginalPixel2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_1 = cv2.resize(frame_1, (LONGITUDE, LONGITUDE))
        frame_1 = np.array(frame_1).reshape(-1)
        label.append(frame_1)
    for i in range(batch_size):
        frame_2 = cv2.imread('./cropedoriginalUS2/%d.jpg' % (n_pic + batch_size * (i+1) ), 0)
        frame_2 = cv2.resize(frame_2, (LONGITUDE, LONGITUDE))
        frame_2 = np.array(frame_2).reshape(-1)
        label_0.append(frame_2)
    return image , label , label_0

x = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
y = tf.placeholder(tf.float32, [None, n_outputs])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
# def max_pool_2x2(x):
#   return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def max_pool_2x2x1(x):
  return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],strides=[1, 1, 2, 2, 1], padding='SAME')

def add_conv(tensor,input_height,output_height,frame_size=CONV_FRAME):
    W_conv=weight_variable([5,frame_size,frame_size,input_height,output_height])
    b_conv=bias_variable([output_height])
    x_image=tf.reshape(tensor,[-1,LONGITUDE,WIDTH,input_height])
    h_conv= tf.nn.relu(conv3d(x_image, W_conv) + b_conv)
    h_pool = max_pool_2x2x1(h_conv)
    return h_pool