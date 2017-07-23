import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
#import self_coded_defs

#load data
def gray2binary(a):
    for i in range(len(a)):
        if a[i]>127:
            a[i]=1
        elif a[i]<=127:
            a[i]=0
    return a

def get_batch(batch_size=20,data_size=6498):
    ran = np.random.choice(data_size, batch_size,replace=False)
    image=[]
    outline=[]
    for i in range(batch_size):
        n_pic=ran[i]
        #print(n_pic)
        frame_0 = cv2.imread('./easyPixelImage2/%d.jpg' % n_pic,0)
        frame_0 = cv2.resize(frame_0, (24, 24))
        frame_0 = np.array(frame_0).reshape(-1)
        # print('np',frame_0)
        # frame_0 = gray2binary(frame_0)
        #print (frame_0)
        frame_1 = cv2.imread('./easyPixelImage2/%d.jpg' % n_pic, 0)
        frame_1 = cv2.resize(frame_1, (24, 24))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        image.append(frame_0)
        outline.append(frame_1)
        #print(image)
    return np.array(image),np.array(outline)


x = tf.placeholder("float", shape=[None, 24*24])
y = tf.placeholder("float", shape=[None, 24*24])
W = tf.Variable(tf.random_uniform(shape=[576,576], minval=0.00001,maxval=0.001,dtype=tf.float32))
b = tf.Variable(tf.random_uniform(shape=[576], minval=0.00001,maxval=0.001,dtype=tf.float32))

init=tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.001, shape=shape)
  return tf.Variable(initial)

def add_full_connect(tensor,input_height,output_height):
    W_conv = weight_variable([input_height, output_height])
    b_conv = bias_variable([output_height])
    tensor_flat=tf.reshape(tensor,[-1,input_height])
    h_fc=tf.nn.tanh(tf.matmul(tensor_flat,W_conv)+b_conv)
    return h_fc

#x_image=tf.reshape(x,[-1,24*24])
h_1 = add_full_connect(x,24*24,256)
h_2 = add_full_connect(h_1,256,64)
h_m = add_full_connect(h_2,64,5)
h_m_2=add_full_connect(h_m,5,64)
h_3 = add_full_connect(h_m_2,64,256)
h_4 = add_full_connect(h_3,256,24*24)

sess.run(tf.initialize_all_variables())

# def cross_entropy(x):
#     return -x*tf.log(x)-(1-x)*tf.log(1-x)

#cost=tf.reduce_sum(tf.square(tf.reshape(h_4,[-1])-tf.reshape(y, [-1])))
cost = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(h_4,[-1])/255,labels=tf.reshape(y, [-1])/255)
train_step = tf.train.AdamOptimizer(0.0005).minimize(cost)

#start training

# xs = get_batch
# ys = xs
with tf.Session() as sess:
    #sess.run(init)
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        xs,ys = get_batch()

        sess.run(train_step, feed_dict={x: xs, y: ys})
        if i%10==0:
            loss=cost.eval(feed_dict={x:xs,y:ys})
            print('batch ' ,(i+10) , ' loss = ' , loss)


    xs,ys = get_batch()
    encode_decode = sess.run(h_4, feed_dict={x: xs,y: ys})
    # Compare original images with their reconstructions
    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(10):
        a[0][i].imshow(np.reshape(xs[i], (24, 24)))
        a[1][i].imshow(np.reshape(ys[i], (24, 24)))
        a[2][i].imshow(np.reshape(encode_decode[i], (24, 24)))
    plt.show()

