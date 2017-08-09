# -*- coding: utf-8 -*-
"""
Calculate the histogram of SIFT descriptors 

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *


data_folder = '../../data/'
set_name_list = ['train','testAB']
fast = cv2.FastFeatureDetector_create()
sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()
trace_save ={}
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

N_centroids = 8
H_ind = 3
set_name_list = ['train','testAB']


des_stack = []

for set_name in set_name_list:
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
        
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})
    for ind,value in SAM_INFO.iterrows():
        print(set_name +'__' +  str(ind) )
        sli_info = pic_sample[(pic_sample.SAM_ID == value.SAM_ID)&(pic_sample.TIM_ID == value.TIM_ID)]
        for ind2,value2 in sli_info.head(1).iterrows():       
            pic_ind = value2.PIC_IND   
            sample_ind = value2.SAM_ID
            row_id = value2.ROW_ID2
            col_id = value2.COL_ID2 
            T_ind = 15 + value.TIM_ID
            if set_name =='train':
                mat = read_sample_trn(input_file, input_size,sample_ind,T_ind,H_ind) 
            if set_name =='testAB':
                mat = read_sample_AB(input_file, input_size,sample_ind,T_ind,H_ind)          
            
            kp = fast.detect(mat,None)
            kp, des = sift.compute(mat,kp)
            if ind==0:
                des_stack = des[:]
            else:
                des_stack = np.vstack([des_stack,des])
                
k_means = KMeans(init='k-means++', n_clusters=N_centroids, n_init=10,random_state=23)
k_means.fit(des_stack)
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)

#%% 

kp_name = map(lambda x:'kp'+ str(x).zfill(2), np.arange(1,N_centroids+1))
hist_bin = np.arange(-0.5,N_centroids+0.5)


for set_name in set_name_list:
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
        
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})
    for ind,value in SAM_INFO.iterrows():
        sli_info = pic_sample[(pic_sample.SAM_ID == value.SAM_ID)&(pic_sample.TIM_ID == value.TIM_ID)]
        for ind2,value2 in sli_info.head(1).iterrows():       
            pic_ind = value2.PIC_IND   
            sample_ind = value2.SAM_ID
            row_id = value2.ROW_ID2
            col_id = value2.COL_ID2 
            T_ind = 15 + value.TIM_ID
            if set_name =='train':
                mat = read_sample_trn(input_file, input_size,sample_ind,T_ind,H_ind) 
            if set_name =='testAB':
                mat = read_sample_AB(input_file, input_size,sample_ind,T_ind,H_ind)   
            kp, des = sift.compute(mat,kp)
            des_class = pairwise_distances_argmin(des, k_means_cluster_centers)
            des_hist = np.histogram(des_class,hist_bin)[0]
        for ind2,value2 in sli_info.iterrows():  
            for i,kp_name_ind in enumerate(kp_name):
                pic_sample.loc[ind2,kp_name_ind] = des_hist[i]
    
    pic_sample[['PIC_IND'] + kp_name].to_csv(data_folder + set_name + '_sift_vector.csv',index = False)

#train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
#train_label['PIC_IND'] = train_label.index + 1
#pic_sample = pd.merge(pic_sample, train_label, how = 'left',on = 'PIC_IND')
#
#feature_list = kp_name
#
#lr_model(pic_sample,feature_list)
#    print(tt4)
##
#np.histogram(tt3,np.arange(10))
        
##tt2 = k_means.fit_transform(des)
#
#tt3 = pairwise_distances_argmin(des, k_means_cluster_centers)
#
#tt4 = np.histogram(tt3,np.arange(-0.5,10.5))
