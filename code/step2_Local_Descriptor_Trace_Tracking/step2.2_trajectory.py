# -*- coding: utf-8 -*-
"""
Calculate the extrapolated trajectory at each of the target site

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
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

T_tar1 = 14
T_tar2 = 15
lim_cri = 20
N_neighbour = 9


#%%
for set_name  in set_name_list:
    
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')
    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
    
    for  H_ind in np.arange(2,5):
		trace_save ={}

        for ind,value in pic_sample.iterrows():
            print(set_name + '_ height:' + str(H_ind ) + ' _pic_ind:' +  str(value.PIC_IND).zfill(5))
            pic_ind = value.PIC_IND   
            sample_ind = value.SAM_ID
            row_id = value.ROW_ID2
            col_id = value.COL_ID2
            time_max = value.TIM_MAX
            T_ind1 = T_tar1 + value.TIM_ID
            T_ind2 = T_tar2 + value.TIM_ID
            if T_ind2 <= time_max:
                if set_name =='train':
                    mat_t1 = read_sample_trn(input_file, input_size,sample_ind,T_ind1,H_ind) 
                    mat_t2 = read_sample_trn(input_file, input_size,sample_ind,T_ind2,H_ind) 
                if set_name =='testAB':
                    mat_t1 = read_sample_AB(input_file, input_size,sample_ind,T_ind1,H_ind) 
                    mat_t2 = read_sample_AB(input_file, input_size,sample_ind,T_ind2,H_ind)             
        
            kp1 = fast.detect(mat_t1,None)            
            kp2 = fast.detect(mat_t2,None)

            if ((len(kp1)<5)|(len(kp2)<5)):
                trace_list = []
                for N_iter in range(16):
                    trace_list.append([row_id,col_id])
            else:
    
                kp1, des1 = sift.compute(mat_t1,kp1)
                locs1 = [(lambda x:x.pt) (x) for x in kp1]
                des1 = des1.astype( np.uint8 )
                    
                kp2, des2 = sift.compute(mat_t2,kp2)
                locs2 = [(lambda x:x.pt) (x) for x in kp2]
                des2 = des2.astype( np.uint8 )
            
                matches = bf.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)
                good_match = match_twosided(des1,des2)      
            
                src_pts= [ locs1[i] for i in np.where(good_match>0)[0]]
                dst_pts = [ locs2[i] for i in good_match[good_match>0]]
                src_pts = np.asarray(src_pts)
                dst_pts = np.asarray(dst_pts)
                
                disp = np.asarray(dst_pts) - np.asarray(src_pts)
            
            #%%
            
                remove_ind  =  np.where(np.sum(abs(disp),axis = 1) <lim_cri)
                src_pts = src_pts[remove_ind  ]
                dst_pts = dst_pts[remove_ind ]
                disp = disp[remove_ind  ]
            
                
            #%% find nearest point in dst_pts[row_id, col_id]
            
                trace_list = []
                for N_iter in range(16):
                    trace_list.append([row_id,col_id])
                    dist = np.sqrt(np.sum(([row_id,col_id] - dst_pts)**2,axis = 1) )
                    useful_index = dist.argsort()[0:N_neighbour]
                    dist[useful_index]
                    disp[useful_index]
                    [row_shift,col_shift] = np.median(disp[useful_index],axis=0)
                    if ((math.isnan(row_shift))|(math.isnan(col_shift)) ):
                        row_shift,col_shift = 0,0
                    row_id = row_id - row_shift
                    col_id = col_id - col_shift
                trace_list = np.asarray(trace_list,dtype='int')
        
#            print(trace_list)
            trace_save[pic_ind] = trace_list
            
        np.save(data_folder+ set_name + '_trace_H'+str(H_ind)+'.npy', trace_save) 
#%%

#read_dictionary = np.load('train_trace_H2.npy').item()


