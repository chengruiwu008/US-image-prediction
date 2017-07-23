import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def get_next_batch():
#
#
#     return xs,ys

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

sess = tf.Session()
x_CNN = tf.placeholder(tf.float32, shape=[None, PIXEL])
x_RNN = tf.placeholder(tf.float32, shape=[None, TIMESTEP, PIXEL])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT])
# W = tf.Variable(tf.random_uniform(shape=[PIXEL,OUTPUT], minval=0.00001,maxval=0.001,dtype=tf.float32))
# b = tf.Variable(tf.random_uniform(shape=[OUTPUT], minval=0.00001,maxval=0.001,dtype=tf.float32))
# sess.run(tf.global_variables_initializer())

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def add_CNN_pool(tensor,input_height,output_height,frame_size=CONV_FRAME):
    W_conv=weight_variable([frame_size,frame_size,input_height,output_height])
    b_conv=bias_variable([output_height])
    x_image=tf.reshape(tensor,[-1,LONGITUDE,WIDTH,1])
    h_conv= tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
    return h_pool

def add_full_connect(tensor,input_height,output_height):
    W_conv = weight_variable([input_height, output_height])
    b_conv = bias_variable([output_height])
    tensor_flat=tf.reshape(tensor,[-1,input_height])
    h_fc=tf.nn.relu(tf.matmul(tensor_flat,W_conv)+b_conv)
    return h_fc

def add_RNN(tensor,n_hidden_units,n_steps=TIMESTEP,batch_size=BATCHSIZE):
    tensor=tf.reshape(tensor,[-1, n_steps, n_hidden_units])
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, tensor,
                                             initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    print(outputs.shape())
    return outputs[-1]

def add_BiLSTM(tensor,n_hidden_units,n_steps=TIMESTEP,batch_size=BATCHSIZE):
    tensor = tf.reshape(tensor, [-1,n_hidden_units])
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_fw_cell ,lstm_bw_cell ,tensor ,dtype=tf.float32)
    return outputs[-1]

def regretion_loss(outputs,target_y,batch_size=BATCHSIZE,n_steps=TIMESTEP):
    target_y=tf.reshape(target_y, [-1,LONGITUDE, WIDTH])
    outputs=tf.reshape(outputs,[-1,LONGITUDE, WIDTH])
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(outputs, [-1], name='reshape_pred')],
            [tf.reshape(target_y, [-1], name='reshape_target')],
            [tf.ones([batch_size * n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=tf.square(tf.subtract(target_y, outputs)),
            name='losses')
    with tf.name_scope('average_cost'):
        cost = tf.div(
            tf.reduce_sum(losses, name='losses_sum'),batch_size,name='average_cost')
    return cost

def train_op(LR=0.0001,train_size=TRAINSIZE):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_op=tf.train.AdamOptimizer(LR).minimize(regretion_loss)
    xs,ys = get_next_batch()
    feed_dict={ x_RNN : xs,y_: ys}

    # plt.ion()
    # plt.show()
    for i in range(train_size):
        if i%10==0:
            loss= sess.run(regretion_loss,feed_dict=feed_dict)
            print('batch ' ,i , ' loss= ' , loss)
            f, a = plt.subplots(2, 1)
            # a[0][1].imshow(np.reshape(image_p[-1], (LONGITUDE, WIDTH)))
            # a[1][1].imshow(np.reshape(res[0], (LONGITUDE, WIDTH)))
            # plt.show()

        sess.run(train_op,feed_dict=feed_dict)


