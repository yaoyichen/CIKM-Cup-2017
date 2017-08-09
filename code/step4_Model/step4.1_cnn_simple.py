# -*- coding: utf-8 -*-
"""
Convolutional neural network training

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *

import tensorflow as tf
import numpy as np
import pandas as pd

data_folder = '../../data/'
    
trn_flat = np.load(data_folder + 'train' + '_flat.npy')
tst_flat = np.load(data_folder + 'testAB' + '_flat.npy')
trn_img = np.load(data_folder + 'train'+ '_3layer_image.npy')
tst_img = np.load(data_folder + 'testAB'+ '_3layer_image.npy')

label = pd.read_csv(data_folder + 'train_image_PICIND.csv')
tst_pic =  pd.read_csv(data_folder + 'testAB_image_PICIND.csv')

trn_img  = 2e-4*trn_img**2
tst_img  = 2e-4*tst_img**2

Y = np.reshape(label.value.values,[-1,1])
N_trn_samples = trn_img.shape[0]


def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0/(fan_in + fan_out))
	high = constant * np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in,fan_out), minval = low, maxval  = high, dtype = tf.float32 )
	
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,seed=12)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def bias_variable_out(shape):
    initial = tf.constant(11.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, 41,41,3])
keep_prob = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob)
#x = tf.placeholder(tf.float32, [None, 121])
#x_image = tf.reshape(x, [-1,11,11,3])
## 31-> 28 -> 14 ->10 -> 5 
## 41-> 39 -> 13 ->12 -> 3
y_true = tf.placeholder(tf.float32, [None,1])

W_conv1 = weight_variable([3,3,3,6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) 
h_pool1 = max_pool_3x3(h_conv1)
h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

W_conv2 = weight_variable([2,2,6,12])
b_conv2 = bias_variable([12])
h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2) 
h_pool2 = max_pool_4x4(h_conv2)
#h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)


W_fc1 = weight_variable([3*3*12 + 49, 30])
b_fc1 = bias_variable([30])


h_flat = tf.reshape(h_pool2, [-1,3*3*12])
#h_flat_drop = tf.nn.dropout(h_flat, keep_prob)
#h_flat_sigmoid = tf.nn.sigmoid(h_flat)

features = tf.placeholder(tf.float32,  [None,49])
h_flat_features = tf.concat([h_flat,features],1)
h_flat_features_drop = tf.nn.dropout(h_flat_features, 0.8)


h_fc1 = tf.nn.relu(tf.matmul(h_flat_features, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#W_fc2 = weight_variable([30, 15])
#b_fc2 = bias_variable([15])
#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([30, 1]) 
b_fc3 = bias_variable_out([1])

y_pred = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

cost = tf.reduce_sum(tf.pow(y_pred-y_true, 2))

train_step = tf.train.AdamOptimizer(1.5e-3).minimize(cost)

#correct_prediction = tf.equal(tf.arg_max(y_conv,1), tf.arg_max(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
iter_list = []
train_loss_list = []
val_loss_list = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
#    split_ind = 4342
    for i in range(1201):
#        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_img[0:split_ind,:,:,:],y_true:Y[0:split_ind], features: trn_flat[0:split_ind],keep_prob:0.65})
#        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_img[split_ind::,:,:,:],y_true:Y[split_ind::], features: trn_flat[split_ind::],keep_prob:0.55})
        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_img,y_true:Y, features: trn_flat,keep_prob:0.65})
        if i%50 ==0:
            train_loss = (cost.eval(feed_dict = {x:trn_img[0:split_ind,:,:,:],y_true:Y[0:split_ind],features: trn_flat[0:split_ind], keep_prob:1.0})/len(Y[0:split_ind]))**0.5
            val_loss = (cost.eval(feed_dict = {x:trn_img[split_ind::,:,:,:],y_true:Y[split_ind::],features: trn_flat[split_ind::], keep_prob:1.0})/len(Y[split_ind::]))**0.5
            iter_list.append(i)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print('step %d, training mse loss %g, val mse loss %g'%(i, train_loss, val_loss))           
        
    test_result = sess.run([y_pred],feed_dict = {x:tst_img[:,:,:,:],features:tst_flat, keep_prob:1.0})
#
tst_pic['value'] = test_result[0]
print(np.mean(test_result[0]))
tst_pic.to_csv(data_folder + 'result_cnn.csv',index = False)
valid_curve = pd.DataFrame({'iter':iter_list,'train_loss':train_loss_list,'val_loss':val_loss_list})
valid_curve.to_csv(data_folder + 'result_cnn_curve.csv',index = False)

