# -*- coding: utf-8 -*-
"""
Stitch images by cross-search among testA and testB set

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *


data_folder = '../../data/'

testB = 'testB'
testB_N_slice = 193
testB_slice_size = pd.read_csv(data_folder + testB + '_slice_size.csv')
testB_input_file = data_folder + testB + '_slice_data'

testA = 'testA'
testA_N_slice = 195
testA_slice_size = pd.read_csv(data_folder + testA + '_slice_size.csv')
testA_input_file = data_folder + testA + '_slice_data'

N_pad = 4
N_pixel = (N_pad*2+1)**2
     
match_all = []
N_match = 0
for slice_id1 in range(1, testA_N_slice+1):
    print('testAB matching: slice id:'+ str(slice_id1).zfill(4))
    search_list =  range(max(slice_id1-7,1) ,min((slice_id1 + 7),testB_N_slice +1  ) ) 
    for T_id1 in range(1,16):
        for H_id in range(1,5):
        
            img1,N_row,N_col = read_slice( testA_input_file ,testA_slice_size, slice_id1,T_id1,H_id)
            for row_cent1 in range(N_pad,N_row -1 - N_pad,30):
                for col_cent1 in range(N_pad,N_col -1 - N_pad,30):
        
                    img_temp = img1[(row_cent1-N_pad): (row_cent1+N_pad+1), (col_cent1-N_pad): (col_cent1+N_pad+1) ]
                    
                    if (np.sum([img_temp ==200] ) == 0)&(np.sum([img_temp > 1] ) >= 0.2*N_pixel):
                        for slice_id2 in search_list:
                            T_id2 = T_id1 
                            img2,N_row,N_col = read_slice(testB_input_file ,testB_slice_size,slice_id2,T_id2,H_id)
                            match_result = cv2_based(img2,img_temp)
                            if len(match_result[0])==1:
                                search_list.remove(slice_id2)
                                row_cent2 = match_result[0][0]+ N_pad
                                col_cent2 = match_result[1][0]+ N_pad
                                match_all.append( (slice_id1, row_cent1, col_cent1,T_id1, slice_id2 , row_cent2, col_cent2,T_id2) )
                                N_match = N_match + 1
                    
                    
match_all_pd = pd.DataFrame(match_all)
match_all_pd.columns = ['testA_SLI_ID','testA_ROW_ID','testA_COL_ID','testA_TIM_ID','testB_SLI_ID','testB_ROW_ID','testB_COL_ID','testB_TIM_ID']
match_all_pd = match_all_pd[['testA_SLI_ID','testA_ROW_ID','testA_COL_ID','testB_SLI_ID','testB_ROW_ID','testB_COL_ID']]

match_all_pd.to_csv(data_folder + 'testAB_SLICE_MATCH.csv',index = False)

testA_slice_size = pd.read_csv(data_folder + 'testA_slice_size.csv')
testB_slice_size = pd.read_csv(data_folder + 'testB_slice_size.csv')
testA_input_file = data_folder +'testA'+'_slice_data'
testB_input_file = data_folder +'testB'+'_slice_data'


testA_MATCH = pd.read_csv(data_folder + 'testA'+ '_TIME_MATCH.csv')
testB_MATCH = pd.read_csv(data_folder + 'testB'+ '_TIME_MATCH.csv')
testAB_SLICE_MATCH = pd.read_csv(data_folder + 'testAB_SLICE_MATCH.csv')
testAB_SLICE_MATCH['ROW_SHIFT'] = testAB_SLICE_MATCH['testB_ROW_ID']- testAB_SLICE_MATCH['testA_ROW_ID']
testAB_SLICE_MATCH['COL_SHIFT'] = testAB_SLICE_MATCH['testB_COL_ID']- testAB_SLICE_MATCH['testA_COL_ID']
testAB_SLICE_MATCH = testAB_SLICE_MATCH[['testA_SLI_ID','testB_SLI_ID','ROW_SHIFT','COL_SHIFT']]
testA_SLI_size = pd.read_csv(data_folder + 'testA_slice_size.csv')
testB_SLI_size = pd.read_csv(data_folder + 'testB_slice_size.csv')

testA_SLI_size.columns = ['SLI_ID','ROW_SIZE','COL_SIZE','start_pos']
testB_SLI_size.columns = ['SLI_ID','ROW_SIZE','COL_SIZE','start_pos']

testA_MATCH = pd.merge(testA_MATCH ,testA_SLI_size, on = ['SLI_ID'],how = 'left')
testB_MATCH = pd.merge(testB_MATCH ,testB_SLI_size, on = ['SLI_ID'],how = 'left')
testB_MATCH['testB_row_sta'] = testB_MATCH['ROW_ID']
testB_MATCH['testB_row_end'] = testB_MATCH['ROW_ID'] + testB_MATCH['ROW_SIZE'] -1
testB_MATCH['testB_col_sta'] = testB_MATCH['COL_ID']
testB_MATCH['testB_col_end'] = testB_MATCH['COL_ID'] + testB_MATCH['COL_SIZE'] -1


testA_MATCH['testA_row_sta'] = testA_MATCH['ROW_ID']
testA_MATCH['testA_row_end'] = testA_MATCH['ROW_ID'] + testA_MATCH['ROW_SIZE'] -1
testA_MATCH['testA_col_sta'] = testA_MATCH['COL_ID']
testA_MATCH['testA_col_end'] = testA_MATCH['COL_ID'] + testA_MATCH['COL_SIZE'] -1
#for sam_id in range(1,1+testB_MATCH.SAM_ID.max()):
#%%

output_file = data_folder + 'testAB'+'_sample_data'
f = open(output_file, "w")
f.close()
sample_stat =[]
size_all = 0
for sam_id in range(1,1+testB_MATCH.SAM_ID.max()):
    print('testAB matiching sample id: ' + str(sam_id).zfill(4) )

    SAMSB = testB_MATCH[testB_MATCH.SAM_ID == sam_id]

    testAB_match = testAB_SLICE_MATCH[testAB_SLICE_MATCH.testB_SLI_ID.isin(SAMSB.SLI_ID)]
    if len(testAB_match)>0:
        SAMSA_part = testA_MATCH[testA_MATCH['SLI_ID'].isin(testAB_match.testA_SLI_ID)]
        SAMSA = testA_MATCH[testA_MATCH.SAM_ID == SAMSA_part.SAM_ID.unique()[0]]
        
        SAMSA = SAMSA.rename(columns = {'SLI_ID':'testA_SLI_ID', 'TIM_ID':'testA_TIM_ID'})
        SAMSB = SAMSB.rename(columns = {'SLI_ID':'testB_SLI_ID', 'TIM_ID':'testB_TIM_ID'})
        testAB_match = pd.merge(testAB_match,SAMSA[['testA_SLI_ID','testA_row_sta','testA_col_sta','testA_TIM_ID']],on = 'testA_SLI_ID', how = 'left')
        testAB_match = pd.merge(testAB_match,SAMSB[['testB_SLI_ID','testB_row_sta','testB_col_sta','testB_TIM_ID']],on = 'testB_SLI_ID', how = 'left')
        testAB_match['testA_row_sta_sft'] = testAB_match['testB_row_sta'] + testAB_match['ROW_SHIFT']
        testAB_match['testA_col_sta_sft'] = testAB_match['testB_col_sta'] + testAB_match['COL_SHIFT']
        
        testAB_match['testA_sam_row_sft'] = testAB_match['testA_row_sta_sft'] - testAB_match['testA_row_sta']
        testAB_match['testA_sam_col_sft'] = testAB_match['testA_col_sta_sft'] - testAB_match['testA_col_sta']
        testAB_match['testA_sam_tim_sft'] = testAB_match['testB_TIM_ID'] - testAB_match['testA_TIM_ID']
        # obtain the shifted index in samA
        ROW_SHIFT = testAB_match['testA_sam_row_sft'].unique()[0]
        COL_SHIFT = testAB_match['testA_sam_col_sft'].unique()[0]
        TIM_SHIFT = testAB_match['testA_sam_tim_sft'].unique()[0]
        if(TIM_SHIFT!=0):
            print('time shift is not always 0!!!')
        
        SAMSA['testA_row_sta_new'] = SAMSA['testA_row_sta'] + ROW_SHIFT
        SAMSA['testA_row_end_new'] = SAMSA['testA_row_end'] + ROW_SHIFT
        SAMSA['testA_col_sta_new'] = SAMSA['testA_col_sta'] + COL_SHIFT
        SAMSA['testA_col_end_new'] = SAMSA['testA_col_end'] + COL_SHIFT
        
        ROW_MIN = np.min( [SAMSB['testB_row_sta'].min(), SAMSA['testA_row_sta_new'].min()] )
        ROW_MAX = np.max( [SAMSB['testB_row_end'].max(), SAMSA['testA_row_end_new'].max()] )
        COL_MIN = np.min( [SAMSB['testB_col_sta'].min(), SAMSA['testA_col_sta_new'].min()] )
        COL_MAX = np.max( [SAMSB['testB_col_end'].max(), SAMSA['testA_col_end_new'].max()] )
        TIME_MAX = np.max( [SAMSA['TIM_MAX'].max(), SAMSB['TIM_MAX'].max()] )

        ROW_SIZE = ROW_MAX - ROW_MIN + 1
        COL_SIZE = COL_MAX - COL_MIN + 1
        
        ROW_SHIFT_A = ROW_SHIFT - ROW_MIN
        ROW_SHIFT_B = 0 - ROW_MIN
        
        COL_SHIFT_A = COL_SHIFT - COL_MIN
        COL_SHIFT_B = 0 - COL_MIN
        for t_id in range(1,1 + TIME_MAX):
            for H_id in range(1,5):
                TH_ind = (t_id-1)*4 + (H_id - 1)      
                view_all = (200*np.ones([ROW_SIZE, COL_SIZE])).astype(np.ubyte)  
                
                for ind,value in SAMSB.iterrows():
                    sam_id = value.SAM_ID
                    slice_id = value.testB_SLI_ID
                    row_id = value.ROW_ID
                    col_id = value.COL_ID
                    tim_id = value.testB_TIM_ID
                    T_id = t_id - tim_id 
                    if ((T_id>=1)&(T_id<=15)):                  
                        data_mat,row,col = read_slice(testB_input_file,testB_slice_size,slice_id,T_id,H_id)
                        useful_place = np.where(data_mat!=200)
                        view_all[ROW_SHIFT_B+ useful_place[0] + row_id,COL_SHIFT_B +  useful_place[1] + col_id] = data_mat[useful_place]
                if TH_ind ==0:  
                    sample_stat.append((SAMSA.SAM_ID.values[0], SAMSB.SAM_ID.values[0], ROW_SIZE,COL_SIZE,TIME_MAX,size_all,ROW_SHIFT_A,COL_SHIFT_A,ROW_SHIFT_B,COL_SHIFT_B))
                    size_all = size_all + 4*TIME_MAX*ROW_SIZE*COL_SIZE
                for ind,value in SAMSA.iterrows():
                    slice_id = value.testA_SLI_ID
                    row_id = value.ROW_ID
                    col_id = value.COL_ID
                    tim_id = value.testA_TIM_ID
                    T_id = t_id - tim_id 
                    if ((T_id>=1)&(T_id<=15)):                  
                        data_mat,row,col = read_slice(testA_input_file,testA_slice_size,slice_id,T_id,H_id)
                        useful_place = np.where(data_mat!=200)
                        view_all[ROW_SHIFT_A + useful_place[0] + row_id,COL_SHIFT_A +  useful_place[1] + col_id] = data_mat[useful_place]


                f = open(output_file, "a")
                f.write(view_all.tobytes())  
                f.close()  
    else:
        TIME_MAX = SAMSB.TIM_MAX.values[0]
        ROW_SIZE = SAMSB.ROW_MAX.values[0]
        COL_SIZE = SAMSB.COL_MAX.values[0]
        for t_id in range(1,1 + TIME_MAX):
            for H_id in range(1,5):
                TH_ind = (t_id-1)*4 + (H_id - 1)      
                view_all = (200*np.ones([ROW_SIZE, COL_SIZE])).astype(np.ubyte)  
                
                for ind,value in SAMSB.iterrows():
                    sam_id = value.SAM_ID
                    slice_id = value.SLI_ID
                    row_id = value.ROW_ID
                    col_id = value.COL_ID
                    tim_id = value.TIM_ID
                    T_id = t_id - tim_id 
                    if ((T_id>=1)&(T_id<=15)):                  
                        data_mat,row,col = read_slice(testB_input_file,testB_slice_size,slice_id,T_id,H_id)
                        useful_place = np.where(data_mat!=200)
                        view_all[ useful_place[0] + row_id, useful_place[1] + col_id] = data_mat[useful_place] 
                if TH_ind ==0:  
                    sample_stat.append((0, SAMSB.SAM_ID.values[0], ROW_SIZE,COL_SIZE,TIME_MAX,size_all,0,0,0,0))
                    size_all = size_all + 4*TIME_MAX*ROW_SIZE*COL_SIZE                        
                f = open(output_file, "a")
                f.write(view_all.tobytes())  
                f.close()  
sample_stat_pd = pd.DataFrame( sample_stat, columns = ['testA_SAM_ID','testB_SAM_ID','N_row','N_col','N_time','start_pos','testA_ROW_SHIFT','testA_COL_SHIFT','testB_ROW_SHIFT','testB_COL_SHIFT'])
sample_stat_pd.to_csv(data_folder + 'testAB' + '_sample_size.csv',index = False)    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pic_slice = pd.read_csv(data_folder  +'testB' + '_MATCH.csv')
slice_sample = pd.read_csv( data_folder +'testB' + '_TIME_MATCH.csv')
sample_size = pd.read_csv(data_folder + 'testAB' + '_sample_size.csv')
slice_sample = pd.merge(slice_sample, sample_size[['testB_SAM_ID','testB_ROW_SHIFT','testB_COL_SHIFT']] ,how= 'left',left_on = 'SAM_ID',right_on = 'testB_SAM_ID')

pic_sample = pd.merge(pic_slice,slice_sample,how = 'left',left_on = 'SLICE_IND', right_on = 'SLI_ID')
pic_sample['ROW_ID2'] = pic_sample['ROW_ID'] + pic_sample['ROW_IND'] + pic_sample['testB_ROW_SHIFT'] 
pic_sample['COL_ID2'] = pic_sample['COL_ID'] + pic_sample['COL_IND']+  pic_sample['testB_COL_SHIFT'] 

sample_stat_pd = pd.read_csv(data_folder + 'testAB' + '_sample_size.csv')  
sample_stat_pd = sample_stat_pd[['testB_SAM_ID','N_row','N_col']]
pic_sample = pd.merge(pic_sample, sample_stat_pd, left_on ='SAM_ID',right_on = 'testB_SAM_ID', how = 'left' )
pic_sample = pic_sample[['PIC_IND','TIM_ID','SAM_ID','TIM_MAX','ROW_ID2','COL_ID2','N_row', 'N_col']]
pic_sample = pic_sample.rename(columns = {'N_row':'ROW_MAX','N_col':'COL_MAX'})
pic_sample.to_csv(data_folder + 'testABB_pic_sample.csv', index = False)

#%%

set_name = 'train'
input_file_sample = data_folder + set_name +'_sample_data'
sample_stat_pd = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
pic_slice = pd.read_csv(data_folder  +set_name + '_MATCH.csv')
slice_sample = pd.read_csv( data_folder +set_name + '_TIME_MATCH.csv')
pic_sample = pd.merge(pic_slice,slice_sample,how = 'left',left_on = 'SLICE_IND', right_on = 'SLI_ID')
pic_sample['ROW_ID2'] = pic_sample['ROW_ID'] + pic_sample['ROW_IND']
pic_sample['COL_ID2'] = pic_sample['COL_ID'] + pic_sample['COL_IND']
pic_sample = pic_sample[['PIC_IND','TIM_ID','SAM_ID','ROW_MAX','COL_MAX','TIM_MAX','ROW_ID2','COL_ID2']]
pic_sample.to_csv(data_folder + 'train_pic_sample.csv', index = False)


#%%
pic_slice = pd.read_csv(data_folder  +'testA' + '_MATCH.csv')
slice_sample = pd.read_csv( data_folder +'testA' + '_TIME_MATCH.csv')
pic_sample = pd.merge(pic_slice,slice_sample, left_on = 'SLICE_IND', right_on = 'SLI_ID',how = 'left')
sample_size = pd.read_csv(data_folder + 'testAB' + '_sample_size.csv')
sample_size = sample_size[['testA_SAM_ID','testB_SAM_ID','testA_ROW_SHIFT','testA_COL_SHIFT']]
pic_sample = pd.merge(pic_sample,sample_size, left_on = 'SAM_ID', right_on = 'testA_SAM_ID',how = 'left' )
pic_sample['ROW_ID2'] = pic_sample['ROW_ID'] + pic_sample['ROW_IND'] + pic_sample['testA_ROW_SHIFT'] 
pic_sample['COL_ID2'] = pic_sample['COL_ID'] + pic_sample['COL_IND']+  pic_sample['testA_COL_SHIFT'] 

del pic_sample['testA_SAM_ID']
del pic_sample['SAM_ID']
pic_sample = pic_sample.rename(columns = {'testB_SAM_ID':'SAM_ID'})
#%%
sample_stat_pd = pd.read_csv(data_folder + 'testAB' + '_sample_size.csv')  
sample_stat_pd = sample_stat_pd[['testB_SAM_ID','N_row','N_col']]
pic_sample = pd.merge(pic_sample, sample_stat_pd, left_on ='SAM_ID',right_on = 'testB_SAM_ID', how = 'left' )
pic_sample = pic_sample[['PIC_IND','TIM_ID','SAM_ID','TIM_MAX','ROW_ID2','COL_ID2','N_row', 'N_col']]
pic_sample = pic_sample.rename(columns = {'N_row':'ROW_MAX','N_col':'COL_MAX'})
#%%
pic_sample = pic_sample[['PIC_IND','TIM_ID','SAM_ID','ROW_MAX','COL_MAX','TIM_MAX','ROW_ID2','COL_ID2']]
pic_sample.to_csv(data_folder + 'testABA_pic_sample.csv', index = False)


