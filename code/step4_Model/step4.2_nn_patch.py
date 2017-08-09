# -*- coding: utf-8 -*-
"""
Neural net training for samples without tracked local image 

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *


data_folder = '../../data/'
    
trn_patch = np.load(data_folder + 'train' + '_patch.npy')
tst_patch = np.load(data_folder + 'testAB' + '_patch.npy')

label = pd.read_csv(data_folder + 'train_patch_PICIND.csv')
tst_pic =  pd.read_csv(data_folder + 'testAB_patch_PICIND.csv')


#trn_patch = np.load(data_folder + 'train' + '_flat.npy')
#tst_patch = np.load(data_folder + 'testAB' + '_flat.npy')
##
#label = pd.read_csv(data_folder + 'train_image_PICIND.csv')
#tst_pic =  pd.read_csv(data_folder + 'testAB_image_PICIND.csv')

Y = np.reshape(label.value.values,[-1,1])
N_trn_samples = trn_patch.shape[0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,seed=23)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def bias_variable_out(shape):
    initial = tf.constant(5.0, shape = shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None,49])
keep_prob = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob)

y_true = tf.placeholder(tf.float32, [None,1])

W_fc1 = weight_variable([49, 10])
b_fc1 = bias_variable([10])
h_fc1 = tf.nn.sigmoid(tf.matmul(x_drop, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#W_fc2 = weight_variable([30, 15])
#b_fc2 = bias_variable([15])
#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([10, 1]) 
b_fc3 = bias_variable_out([1])

y_pred = tf.matmul(h_fc1_drop, W_fc3) + b_fc3
                  
cost = tf.reduce_sum(tf.pow(y_pred-y_true, 2))

train_step = tf.train.AdamOptimizer(2.0e-3).minimize(cost)

iter_list = []
train_loss_list = []
val_loss_list = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    split_ind = 600
    for i in range(2001):
#        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_patch[split_ind::,:],y_true:Y[split_ind::],keep_prob:0.7})
#        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_patch[0:split_ind:,:],y_true:Y[0:split_ind],keep_prob:0.8})
        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_patch,y_true:Y,keep_prob:0.7})
        if i%5 ==0:
            train_loss = (cost.eval(feed_dict = {x:trn_patch[0:split_ind,:,],y_true:Y[0:split_ind], keep_prob:1.0})/len(Y[0:split_ind]))**0.5
            val_loss = (cost.eval(feed_dict = {x:trn_patch[split_ind::,:],y_true:Y[split_ind::], keep_prob:1.0})/len(Y[split_ind::]))**0.5
            iter_list.append(i)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print('step %d, training mse loss %g, val mse loss %g'%(i, train_loss, val_loss))           
        
    test_result = sess.run([y_pred],feed_dict = {x:tst_patch, keep_prob:1.0})
#
tst_pic['value'] = test_result[0]
print(np.mean(test_result[0]))
tst_pic.to_csv(data_folder + 'result_nnpatch.csv',index = False)

valid_curve = pd.DataFrame({'iter':iter_list,'train_loss':train_loss_list,'val_loss':val_loss_list})
valid_curve.to_csv(data_folder + 'result_nnpatch_curve.csv',index = False)
                  
#plt.plot(iter_list,train_loss_list )
#plt.plot(iter_list,val_loss_list )


