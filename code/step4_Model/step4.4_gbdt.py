# -*- coding: utf-8 -*-
"""
GBDT model

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03,max_depth=1, random_state=0)

data_folder = '../../data/'

#%%    
trn_patch = np.load(data_folder + 'train' + '_patch.npy')
tst_patch = np.load(data_folder + 'testAB' + '_patch.npy')
label = pd.read_csv(data_folder + 'train_patch_PICIND.csv')
tst_pic =  pd.read_csv(data_folder + 'testAB_patch_PICIND.csv')

X = trn_patch
Y = label.value.values
X_test = tst_patch

model.fit(X,Y)
Y_pred = model.predict(X)
Y_test_pred = model.predict(X_test)
print( np.std((Y_pred- Y) ) )

Y_test_pred = model.predict(X_test)
tst_pic['value'] = Y_test_pred
       
tst_pic.to_csv(data_folder + 'result_gbdtpatch.csv',index = False)


#%%
trn_patch = np.load(data_folder + 'train' + '_flat.npy')
tst_patch = np.load(data_folder + 'testAB' + '_flat.npy')
label = pd.read_csv(data_folder + 'train_image_PICIND.csv')
tst_pic =  pd.read_csv(data_folder + 'testAB_image_PICIND.csv')
##%%
X = trn_patch
Y = label.value.values
X_test = tst_patch

model.fit(X,Y)
Y_pred = model.predict(X)
Y_test_pred = model.predict(X_test)
print( np.std((Y_pred- Y) ) )

Y_test_pred = model.predict(X_test)
tst_pic['value'] = Y_test_pred
       
tst_pic.to_csv(data_folder + 'result_gbdt.csv',index = False)

#%%
#split_ind = 600
#X = trn_patch[0:split_ind,:]
#Y = label[0:split_ind].value.values
#
#X_test = trn_patch[split_ind::,:]
#Y_test = label[split_ind::].value.values
#
#model.fit(X,Y)
#Y_pred = model.predict(X)
#Y_test_pred = model.predict(X_test)
#print( np.std((Y_pred- Y) ) )
#print( np.std((Y_test_pred- Y_test)) )
#
#
#model.fit(X_test,Y_test)
#Y_pred = model.predict(X)
#Y_test_pred = model.predict(X_test)
#print( np.std((Y_test_pred- Y_test)) )
#print( np.std((Y_pred- Y) ) )




