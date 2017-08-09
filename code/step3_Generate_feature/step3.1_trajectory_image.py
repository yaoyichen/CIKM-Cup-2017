# -*- coding: utf-8 -*-
"""
Generate local image feature at the surroudning area
of the extrapolation time stamp. 

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *
data_folder = '../../data/'

set_name_list = ['train','testAB']
for set_name in set_name_list:
    N_pad = 20
    N_len = 2*N_pad + 1
    N_area = (2*N_pad+1)**2
    N_count = 0
    T_src = 15
    pic_list = []
    data_list = []
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
#        train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
#        train_label['PIC_IND'] = train_label.index + 1
#        pic_sample = pd.merge(pic_sample, train_label, how = 'left',on = 'PIC_IND')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
    
    trace_H2 = np.load(data_folder + set_name + '_trace_H2.npy').item()
    trace_H3 = np.load(data_folder + set_name + '_trace_H3.npy').item()    
    trace_H4 = np.load(data_folder + set_name + '_trace_H4.npy').item()
    
    for ind,value in pic_sample.iterrows():
        PIC_IND = value.PIC_IND
        sample_ind = value.SAM_ID
        row_id_src_H2 = trace_H2[PIC_IND][14][0]
        col_id_src_H2 = trace_H2[PIC_IND][14][1]
        row_id_src_H3 = trace_H3[PIC_IND][14][0]
        col_id_src_H3 = trace_H3[PIC_IND][14][1]
        row_id_src_H4 = trace_H4[PIC_IND][14][0]
        col_id_src_H4 = trace_H4[PIC_IND][14][1]
        T_ind_src = T_src + value.TIM_ID
        if ((row_id_src_H2>=N_pad)&(col_id_src_H2>=N_pad)&(row_id_src_H3>=N_pad)&(col_id_src_H3>=N_pad)&(row_id_src_H4>=N_pad)&(col_id_src_H4>=N_pad)):
            if set_name =='train':
                mat_src_H2 = read_sample_trn(input_file, input_size,sample_ind,T_ind_src,2) 
                mat_src_H3 = read_sample_trn(input_file, input_size,sample_ind,T_ind_src,3)
                mat_src_H4 = read_sample_trn(input_file, input_size,sample_ind,T_ind_src,4)         
            if set_name == 'testAB':
                mat_src_H2 = read_sample_AB(input_file, input_size,sample_ind,T_ind_src,2) 
                mat_src_H3 = read_sample_AB(input_file, input_size,sample_ind,T_ind_src,3)
                mat_src_H4 = read_sample_AB(input_file, input_size,sample_ind,T_ind_src,4) 
            
            mat_src_H2_pad = mat_src_H2[row_id_src_H2-N_pad:row_id_src_H2+N_pad +1,col_id_src_H2-N_pad:col_id_src_H2 +N_pad+1]  
            mat_src_H3_pad = mat_src_H3[row_id_src_H3-N_pad:row_id_src_H3+N_pad +1,col_id_src_H3-N_pad:col_id_src_H3 +N_pad+1] 
            mat_src_H4_pad = mat_src_H4[row_id_src_H4-N_pad:row_id_src_H4+N_pad +1,col_id_src_H4-N_pad:col_id_src_H4 +N_pad+1] 
            
            good_values_src_H2 = np.sum(mat_src_H2_pad!=200)  
            size_src_H2 = np.size(mat_src_H2_pad)
            good_values_src_H3 = np.sum(mat_src_H3_pad!=200)  
            size_src_H3 = np.size(mat_src_H3_pad)
            good_values_src_H4 = np.sum(mat_src_H4_pad!=200)  
            size_src_H4 = np.size(mat_src_H4_pad)
            
            if ((good_values_src_H2>=N_area)&(size_src_H2 == N_area)&(good_values_src_H3>=N_area)&(size_src_H3 == N_area)&(good_values_src_H4>=N_area)&(size_src_H4 == N_area)) :
                pic_list.append(PIC_IND)    
                N_count = N_count +1 
                data_list.append([mat_src_H2_pad ,mat_src_H3_pad ,mat_src_H4_pad ])
    print( N_count)
        
    #%% write data
    data_list_np = np.asarray(data_list, dtype= np.float32)
    data_list_np2 = np.moveaxis(data_list_np,1,3)
    data_list_np3 = data_list_np2.reshape([-1,N_len,N_len,3])
    np.save(data_folder + set_name + '_3layer_image.npy',data_list_np3 )
    feature_list_pd = pd.DataFrame(pic_list, columns = ['PIC_IND'])
    if set_name =='testAB':
        feature_list_pd.to_csv(data_folder + set_name + '_image_PICIND.csv',index = False)
    if set_name =='train':
        train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
        train_label['PIC_IND'] = train_label.index + 1
        
        feature_list_pd = pd.merge(feature_list_pd, train_label, how = 'left',on = 'PIC_IND')
        feature_list_pd.to_csv(data_folder + set_name + '_image_PICIND.csv',index = False)

