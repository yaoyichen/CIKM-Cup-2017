# -*- coding: utf-8 -*-
"""
Generate global description of the cloud pattern 

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
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
        train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
        train_label['PIC_IND'] = train_label.index + 1
        pic_sample = pd.merge(pic_sample, train_label, how = 'left',on = 'PIC_IND')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
        
    #%%  Velocity and accelation vector
        
    velo_ = map(lambda x:'V'+ str(x).zfill(2), np.arange(1,9))
    serie_ = np.arange(14).reshape([14,-1])
    trace_H2 = np.load(data_folder + set_name +'_trace_H2.npy').item()
    model = linear_model.LinearRegression(n_jobs= 4)
    for ind,value in pic_sample.iterrows():
        pic_id = value.PIC_IND
        trace = trace_H2[pic_id]
        trace_pre = np.roll(trace,-2)
        velo = trace[1:-1,:] - trace_pre[1:-1,:]
        pic_sample.loc[ind,'V01'] = velo.mean(axis=0)[0]
        pic_sample.loc[ind,'V02'] = velo.mean(axis=0)[1]
        pic_sample.loc[ind,'V03'] = np.sqrt(np.sum(velo.mean(axis=0)**2))
#        pic_sample.loc[ind,'V04'] = velo.std(axis=0)[0]
#        pic_sample.loc[ind,'V05'] = velo.std(axis=0)[1]
        
        angle = math.atan2(velo.mean(axis=0)[1],velo.mean(axis=0)[0])
        pic_sample.loc[ind,'V04'] = angle              # wind direction
        pic_sample.loc[ind,'V05'] = math.sin(angle)
        pic_sample.loc[ind,'V06'] = math.cos(angle)
    

    #%% coordinate
    coord_ = map(lambda x:'C'+ str(x).zfill(2), np.arange(1,9))
    
    pic_sample['C01'] = pic_sample['ROW_ID2']/pic_sample['ROW_MAX']
    pic_sample['C02'] = pic_sample['COL_ID2']/pic_sample['COL_MAX']
    pic_sample['C03'] = np.abs(pic_sample['C01']-0.5)
    pic_sample['C04'] = np.abs(pic_sample['C02']-0.5)
    pic_sample['C05'] = pic_sample['ROW_MAX']*pic_sample['COL_MAX']
#    pic_sample['C06'] = 1.0*pic_sample['ROW_MAX']/pic_sample['COL_MAX']
#    pic_sample['C07'] = np.sqrt(pic_sample['C01']**2 + pic_sample['C01']**2)
#    pic_sample['C08'] = np.sqrt((pic_sample['ROW_ID2']-pic_sample['ROW_MAX'])**2 + (pic_sample['COL_ID2']-pic_sample['COL_MAX'])**2)

    #%% histgram of SIFT descriptor classes
    N_centroids = 8
    kp_ = map(lambda x:'kp'+ str(x).zfill(2), np.arange(1,N_centroids+1)) 
    kp_feature = pd.read_csv(data_folder + set_name + '_sift_vector.csv')
    pic_sample = pd.merge(pic_sample, kp_feature, on = 'PIC_IND', how = 'left')
    
    #%% historical info
    hist_ = map(lambda x:'H'+ str(x).zfill(2), np.arange(1,4))
    pic_sample['H01'] = pic_sample['TIM_ID']
    # T02 whether appear or disappear
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})
    for ind,value in SAM_INFO.iterrows():
        sample_ind = value.SAM_ID
        sli_info = pic_sample[(pic_sample.SAM_ID == sample_ind)&(pic_sample.TIM_ID == value.TIM_ID)]   
        if set_name == 'train':
            mat0 = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+1,2) 
            mat1 = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+6,2) 
            mat2 = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+11,2) 
        if set_name == 'testAB':
            mat0 = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+1,2) 
            mat1 = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+6,2) 
            mat2 = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+11,2) 
        mat0_num = np.sum(mat0!=200)
        mat1_num = np.sum(mat1!=200)
        mat2_num = np.sum(mat2!=200)       
        for ind2,value2 in sli_info.iterrows():
            if mat1_num > mat0_num:
                pic_sample.loc[ind2,'H02'] = 1
            elif(mat1_num == mat0_num):
                pic_sample.loc[ind2,'H02'] = 0
            else:
                pic_sample.loc[ind2,'H02'] = -1             
            if mat2_num > mat1_num:
                pic_sample.loc[ind2,'H03'] = 1
            elif(mat2_num == mat1_num):
                pic_sample.loc[ind2,'H03'] = 0
            else:
                pic_sample.loc[ind2,'H03'] = -1
#%%  statistics on the trajectory
    if set_name =='train':
        time_trace = 15
        N_pad = 20
        N_len = 2*N_pad + 1
        img_hist_H2 = np.zeros(200)
        img_hist_num = np.arange(200)
        SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
        SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})
        for ind,value in SAM_INFO.iterrows():
            sli_info = pic_sample[(pic_sample.SAM_ID == value.SAM_ID)&(pic_sample.TIM_ID == value.TIM_ID)]
            sample_ind = value.SAM_ID
            if set_name =='train':
                img_mat = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+time_trace,2)    
            if set_name == 'testAB':
                img_mat = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+time_trace,2)    
            for i in range(200):
                img_hist_H2[i] = img_hist_H2[i] + np.sum(img_mat==i)               
        cumsum_list = np.cumsum(img_hist_H2)/np.sum(img_hist_H2)
        pct_range = 0.1*np.arange(1,10)
        break_point = []
        for pct_ind in pct_range:
            break_point.append(np.max(np.where(cumsum_list < pct_ind)))
        break_point = list(set(break_point))
        break_point.sort()
        break_point.append(200)
    
      
    bin_ = map(lambda x:'B'+ str(x).zfill(2), np.arange(1,8))   
    SAM_INFO = pic_sample[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'}) 
    trace_H2 = np.load(data_folder + set_name + '_trace_H2.npy').item()
    time_left = (time_trace<<1)
    for ind,value in SAM_INFO.iterrows():
        sli_info = pic_sample[(pic_sample.SAM_ID == value.SAM_ID)&(pic_sample.TIM_ID == value.TIM_ID)]
        sample_ind = value.SAM_ID
        if set_name =='train':
            img_mat = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+15,2)    
        if set_name == 'testAB':
            img_mat = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+15,2)
    
        for ind2,value2 in sli_info.iterrows():
            row_id = trace_H2[value2.PIC_IND][15,0]
            col_id = trace_H2[value2.PIC_IND][15,1]
            img_sub = img_mat[row_id-N_pad:row_id+N_pad +1,col_id-N_pad:col_id +N_pad+1] 
            img_sub_valid = img_sub[img_sub!=200]
            img_sub_valid2 = img_sub[(img_sub!=200)&(img_sub>1)]
            
            row_id0,col_id0 = int(value2.ROW_ID2), int(value2.COL_ID2)
            img_sub0 = img_mat[row_id0-N_pad:row_id0+N_pad +1,col_id0-N_pad:col_id0 +N_pad+1] 
            
            img_sub0_valid = img_sub0[img_sub0!=200]
            img_sub0_valid2 = img_sub0[(img_sub0!=200)]
            
            if len(img_sub_valid)==0:
                pic_sample.loc[ind2,'M01'] = np.max(0.5*img_sub0_valid)**2.0
                pic_sample.loc[ind2,'M02'] = np.median(0.8*img_sub0_valid)**2.0
                pic_sample.loc[ind2,'M03'] = np.mean(0.9*img_sub0_valid)**2.0
                for i in range(len(bin_)):
                    pic_sample.loc[ind2,bin_[i]] = 1.0*np.sum(((0.9*img_sub0_valid)>=break_point[i])&((0.9*img_sub0_valid)<break_point[i+1]))
            else:
                pic_sample.loc[ind2,'M01'] = np.max(img_sub_valid)**2.0
                pic_sample.loc[ind2,'M02'] = np.median(img_sub_valid)**2.0
                pic_sample.loc[ind2,'M03'] = np.mean(img_sub_valid)**2.0
                for i in range(len(bin_)):
                    pic_sample.loc[ind2,bin_[i]] = 1.0*np.sum((img_sub_valid>=break_point[i])&(img_sub_valid<break_point[i+1]))
                              
#        N_pad = 5
#        if sli_info.TIM_ID.values[0] +time_left <= sli_info.TIM_MAX.values[0]:
#            if set_name =='train':
#                img_mat = read_sample_trn(input_file, input_size,sample_ind,value.TIM_ID+ time_left,2)    
#            if set_name == 'testAB':
#                img_mat = read_sample_AB(input_file, input_size,sample_ind,value.TIM_ID+ time_left,2)   
#            
#            for ind2,value2 in sli_info.iterrows():
#                row_id,col_id = int(value2.ROW_ID2), int(value2.COL_ID2)
#                img_sub = img_mat[row_id-N_pad:row_id+N_pad +1,col_id-N_pad:col_id +N_pad+1] 
#                img_sub_valid = img_sub[img_sub!=200]
#                if len(img_sub_valid)!=0:
#                    pic_sample.loc[ind2,'M04'] = np.max(img_sub_valid)**2.0
#                    pic_sample.loc[ind2,'M05'] = 1
#    pic_sample['M04'] = pic_sample['M04'].fillna(pic_sample['M04'].mean()*0.75)
#    pic_sample['M05'] = pic_sample['M05'].fillna(0)         
#%%           
    for ind,value in SAM_INFO.iterrows():
        sli_info = pic_sample[(pic_sample.SAM_ID == value.SAM_ID)&(pic_sample.TIM_ID == value.TIM_ID)]
        argsort = np.argsort(sli_info['M01'].values)
        for i in range(len(argsort)):
            sortid = argsort[i]
            pic_sample.loc[sli_info.index[sortid],'R01'] = i
            pic_sample.loc[sli_info.index[sortid],'R02'] = 1.0*i/len(argsort)   
               
#%% neighbhour information
    if (set_name =='train'):
        pic_sample_temp = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
    if set_name =='testAB':
        pic_sample_A = pd.read_csv(data_folder + set_name+ 'A_pic_sample.csv')
        pic_sample_B = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
        pic_sample_A['SET'] = 'A'
        pic_sample_B['SET'] = 'B'
        pic_sample_temp = pd.concat([pic_sample_A,pic_sample_B])
        
    #%%
    SAM_INFO = pic_sample_temp[['SAM_ID','TIM_ID','PIC_IND']].groupby(['SAM_ID','TIM_ID'],as_index = False).count()
    SAM_INFO = SAM_INFO.rename(columns = {'PIC_IND':'N_sli'})
    N_dist = 10
    Dist_list = (np.e-1)**np.arange(1,N_dist)
    N_ = map(lambda x:'N'+ str(x).zfill(2), np.arange(1,N_dist )) 
#    NP_ = map(lambda x:'NP'+ str(x).zfill(2), np.arange(1,N_num )) 
    dist_zip = zip(Dist_list,N_)
    for ind,value in SAM_INFO.iterrows():
        sli_info = pic_sample_temp[(pic_sample_temp.SAM_ID == value.SAM_ID)&(pic_sample_temp.TIM_ID == value.TIM_ID)]
        for ind2,value2 in sli_info.iterrows():
            row_id = value2.ROW_ID2
            col_id = value2.COL_ID2
            dist_list = np.sqrt((sli_info['ROW_ID2'] - row_id)**2 + (sli_info['COL_ID2'] - col_id)**2)
            for i,[dist,name] in enumerate(dist_zip):
                pic_sample_temp.loc[ind2,name] = np.sum(dist_list<dist) -1
               
    if set_name =='testAB':
        pic_sample_temp = pic_sample_temp[pic_sample_temp['SET']=='B']
        del pic_sample_temp['SET']          
    pic_sample_temp = pic_sample_temp[['PIC_IND'] + N_  ]   
    pic_sample = pd.merge(pic_sample,pic_sample_temp , on = ['PIC_IND'])    
        
    #%%
    velo_ = map(lambda x:'V'+ str(x).zfill(2), np.arange(1,7))
    coord_ = map(lambda x:'C'+ str(x).zfill(2), np.arange(1,6))
    N_centroids = 8
    kp_ = map(lambda x:'kp'+ str(x).zfill(2), np.arange(1,N_centroids+1)) 
    hist_ = map(lambda x:'H'+ str(x).zfill(2), np.arange(1,4))
    
    bin_ = map(lambda x:'B'+ str(x).zfill(2), np.arange(1,8))  
    M_ = map(lambda x:'M'+ str(x).zfill(2), np.arange(1,4)) 
    R_ = map(lambda x:'R'+ str(x).zfill(2), np.arange(1,3)) 
        
    N_dist = 10
    N_ = map(lambda x:'N'+ str(x).zfill(2), np.arange(1,N_dist )) 

    
    pic_sample = pic_sample[['PIC_IND'] + velo_ + coord_ + kp_ +hist_ + bin_ + M_ + R_ + N_ ]
    
    print(pic_sample.describe())
    if set_name =='train':
        pic_sample_train = pic_sample[:]
    if set_name =='testAB':
        pic_sample_testAB = pic_sample[:]        
    pic_sample.to_csv(data_folder + set_name + '_F3.csv',index = False)
    

#%%
#plt.hist(pic_sample_train['B05'],alpha= 0.5)
#plt.hist(pic_sample_testAB['B05'],alpha= 0.3)
    
    




