import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

LONGITUDE=24
WIDTH=24
PIXEL=LONGITUDE*WIDTH
BATCHSIZE=10
TRAINSIZE=10000
CONV_FRAME=3
TIMESTEP=10
CONV1_IN=32
CONV2_IN=64
#CONV3_IN=128
#SOFTMAX_IN=2048
OUTPUT=LONGITUDE*WIDTH

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
        frame_0 = cv2.imread('./cropPicY/%d.jpg' % n_pic,0)
        frame_0 = cv2.resize(frame_0, (24, 24))
        frame_0 = np.array(frame_0).reshape(-1)
        # print('np',frame_0)
        # frame_0 = gray2binary(frame_0)
        #print (frame_0)
        frame_1 = cv2.imread('./cropPicX/%d.jpg' % n_pic, 0)
        frame_1 = cv2.resize(frame_1, (24, 24))
        frame_1 = np.array(frame_1).reshape(-1)
        frame_1 = gray2binary(frame_1)
        image.append(frame_0)
        outline.append(frame_1)
        #print(image)
    return np.array(image),np.array(outline)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# def max_pool_2x2(x):
#   return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def add_conv(tensor,input_height,output_height,frame_size=CONV_FRAME):
    W_conv=weight_variable([frame_size,frame_size,input_height,output_height])
    b_conv=bias_variable([output_height])
    x_image=tf.reshape(tensor,[-1,LONGITUDE,WIDTH,input_height])
    h_conv= tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
    #h_pool = max_pool_2x2(h_conv)
    return h_conv

x = tf.placeholder("float", shape=[None, 24*24])
y = tf.placeholder("float", shape=[24*24,None])

conv_1 = add_conv(x,1,8)
conv_2 = add_conv(conv_1,8,8)
conv_3 = add_conv(conv_2,8,8)
conv_4 = add_conv(conv_3,8,8)
conv_5 = add_conv(conv_4,8,1,frame_size=1)
#print(np.shape(conv_5))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

def correct_prediction(correct_pred=0):
    for i in range(PIXEL):
        correct_prediction = tf.equal(tf.argmax(conv_5[0][i], 1), tf.argmax(y[i], 1))
        correct_pred += correct_prediction
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(conv_5,[-1]), labels=tf.reshape(y,[-1])))
#cost=tf.reduce_sum(tf.square(tf.reshape(conv_5,[-1])-tf.reshape(y, [-1])))
train_step = tf.train.AdamOptimizer(0.0005).minimize(cost)

#start training

# xs = get_batch
# ys = xs
with tf.Session() as sess:
    #sess.run(init)
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        xs,ys = get_batch()
        zs = np.transpose(ys)
        sess.run(train_step, feed_dict={x: ys, y: zs})
        if i%10==0:
            loss=cost.eval(feed_dict={x:ys,y:zs})
            #acc = sess.run(correct_prediction ,feed_dict={x:ys,y:zs})
            print('batch ' ,(i + 10) , ' loss = ' , loss)
            #print('batch ', (i + 10), ' accuracy = ', acc)


    _,ys = get_batch()
    zs = np.transpose(ys)
    encode_decode = sess.run(conv_5, feed_dict={x: ys,y: zs})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(ys[i], (24, 24)))
        a[1][i].imshow(np.reshape(encode_decode[i], (24, 24)))
    plt.show()

