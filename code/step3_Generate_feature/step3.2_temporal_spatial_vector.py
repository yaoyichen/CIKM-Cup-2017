# -*- coding: utf-8 -*-
"""
Generate temporal and altitudinal vector

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *
data_folder = '../../data/'

set_name_list = ['train','testAB']

for set_name in set_name_list:
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    trace_H2 = np.load(data_folder + set_name + '_trace_H2.npy').item()
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
        train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
        train_label['PIC_IND'] = train_label.index + 1
        pic_sample = pd.merge(pic_sample, train_label, how = 'left',on = 'PIC_IND')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
    

    #%% time diff
    
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'}) 
    time_list = np.arange(11,16)
    cover_ = map(lambda x:'COV'+ str(x).zfill(2), time_list) 
    mean_ = map(lambda x:'MEA'+ str(x).zfill(2), time_list) 
    std_ = map(lambda x:'STD'+ str(x).zfill(2), time_list) 
    max_ = map(lambda x:'MAX'+ str(x).zfill(2), time_list) 
    
    item_zip = zip(time_list,cover_, mean_,std_,max_)
    for ind,value in SAM_INFO.iterrows():  
        for t_id,cover_item,mean_item,std_item,max_item in item_zip:   
            if set_name =='train':
                img = read_sample_trn(input_file, input_size,value.SAM_ID,value.TIM_ID+t_id,2)
            else:
                img = read_sample_AB(input_file, input_size,value.SAM_ID,value.TIM_ID+t_id,2)                
            useful_img = img[img!=200]
            none0_img = useful_img[useful_img>5]
            SAM_INFO.loc[ind,cover_item] = 1.0*np.size(none0_img)/np.size(useful_img)
            SAM_INFO.loc[ind,mean_item] = np.mean(useful_img )**2
            SAM_INFO.loc[ind,std_item] = np.std(none0_img)**2
            SAM_INFO.loc[ind,max_item] = np.max(useful_img )**2
            
    pic_sample = pd.merge(pic_sample,SAM_INFO, on = ['SAM_ID','TIM_ID'], how = 'left')      
    
    #%% time diff
    time_diff_list = np.asarray([2,4])
    cover_diff_ = map(lambda x:'COV_DIFF'+ str(x).zfill(2), time_diff_list) 
    mean_diff_ = map(lambda x:'MEA_DIFF'+ str(x).zfill(2), time_diff_list) 
    std_diff_ = map(lambda x:'STD_DIFF'+ str(x).zfill(2), time_diff_list) 
    max_diff_ = map(lambda x:'MAX_DIFF'+ str(x).zfill(2), time_diff_list)   
    
    item_zip = zip(time_diff_list,cover_diff_,mean_diff_,std_diff_,max_diff_)
    for t_id, cover_item, mean_item,std_item,max_item in item_zip:
        pic_sample[cover_item] = 1.0*pic_sample[cover_[t_id]] - pic_sample[cover_[t_id-1]]
        pic_sample[mean_item] = 1.0*pic_sample[mean_[t_id]] - pic_sample[mean_[t_id-1]]
        pic_sample[std_item] = 1.0*pic_sample[std_[t_id]] - pic_sample[std_[t_id-1]]
        pic_sample[max_item] = 1.0*pic_sample[max_[t_id]] - pic_sample[max_[t_id-1]]
    
    #%% height diff
    height_diff_list = np.asarray([3,4])
    cover_diff_H = map(lambda x:'COV_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    mean_diff_H = map(lambda x:'MEA_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    std_diff_H = map(lambda x:'STD_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    max_diff_H = map(lambda x:'MAX_DIFF_H'+ str(x).zfill(2), height_diff_list)     
    
    item_zip = zip(height_diff_list, cover_diff_H,mean_diff_H ,std_diff_H ,max_diff_H)
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})     

    for ind,value in SAM_INFO.iterrows():  
        for h_id,cover_,mean_,std_,max_ in item_zip:     
            if set_name =='train':
                img1 = read_sample_trn(input_file, input_size,value.SAM_ID,value.TIM_ID+15,h_id)
                img2 = read_sample_trn(input_file, input_size,value.SAM_ID,value.TIM_ID+15,h_id-1)
            if set_name =='testAB':
                img1 = read_sample_AB(input_file, input_size,value.SAM_ID,value.TIM_ID+15,h_id)
                img2 = read_sample_AB(input_file, input_size,value.SAM_ID,value.TIM_ID+15,h_id-1)
                
            useful_img1 = img1[img1!=200]
            none0_img1 = useful_img1[useful_img1>5]
            useful_img2 = img2[img2!=200]
            none0_img2 = useful_img2[useful_img2>5]
            SAM_INFO.loc[ind,cover_] = 1.0*np.size(none0_img1)/np.size(useful_img1) - 1.0*np.size(none0_img2)/np.size(useful_img2) 
            SAM_INFO.loc[ind,mean_] = np.mean(useful_img1 )**2 - np.mean(useful_img2 )**2
            if ((len(none0_img2)>0)&(len(none0_img1)>0)):
                SAM_INFO.loc[ind,std_]  = np.std(none0_img1)**2 - np.std(none0_img2)**2 
            else:
                SAM_INFO.loc[ind,std_] = 0.0
            SAM_INFO.loc[ind,max_] = np.max(useful_img1 )**2 - np.max(useful_img2 )**2
            
    pic_sample = pd.merge(pic_sample,SAM_INFO, on = ['SAM_ID','TIM_ID'], how = 'left')
    
#%%
    time_diff_list = np.asarray([2,4])
    cover_diff_ = map(lambda x:'COV_DIFF'+ str(x).zfill(2), time_diff_list) 
    mean_diff_ = map(lambda x:'MEA_DIFF'+ str(x).zfill(2), time_diff_list) 
    std_diff_ = map(lambda x:'STD_DIFF'+ str(x).zfill(2), time_diff_list) 
    max_diff_ = map(lambda x:'MAX_DIFF'+ str(x).zfill(2), time_diff_list)
    height_diff_list = np.asarray([3,4])
    cover_diff_H = map(lambda x:'COV_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    mean_diff_H = map(lambda x:'MEA_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    std_diff_H = map(lambda x:'STD_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    max_diff_H = map(lambda x:'MAX_DIFF_H'+ str(x).zfill(2), height_diff_list) 
    time_list = np.asarray([11,15])
    cover_ = map(lambda x:'COV'+ str(x).zfill(2), time_list) 
    mean_ = map(lambda x:'MEA'+ str(x).zfill(2), time_list) 
    std_ = map(lambda x:'STD'+ str(x).zfill(2), time_list) 
    max_ = map(lambda x:'MAX'+ str(x).zfill(2), time_list) 
    
    TS_ = cover_diff_ + mean_diff_ + std_diff_ + max_diff_ + cover_diff_H + mean_diff_H + std_diff_H + max_diff_H + cover_ + mean_ + std_ + max_
    pic_sample = pic_sample[['PIC_IND'] + TS_]
    print(pic_sample.describe())
    pic_sample.to_csv(data_folder + set_name + '_F2.csv',index = False)
    
