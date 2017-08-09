# -*- coding: utf-8 -*-
"""
Temporal template matching of sub-images

@author: Marmot
"""

import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *

data_folder = '../../data/'
set_name_list = ['train','testA','testB']
N_slice_list = [454,195,193]

N_pad = 4
N_pixel = (N_pad*2+1)**2

time1= time.time()
for set_name,N_slice in zip(set_name_list,N_slice_list):

    slice_size = pd.read_csv(data_folder + set_name + '_slice_size.csv')
    input_file = data_folder + set_name + '_slice_data'
    
    match_all = []
    
    for slice_id1 in range(1, N_slice+1):
        print(set_name + ' slice_ind: ' + str(slice_id1).zfill(4))
        search_list =  range(max(1,slice_id1-2),slice_id1) + range( slice_id1+1,min((slice_id1 + 4),N_slice + 1 ) ) 
        for T_id1 in range(1,16,1):
            for H_id in range(1,5,1):
                img1,N_row,N_col = read_slice(input_file,slice_size,slice_id1,T_id1,H_id)
                for row_cent1 in range(N_pad,N_row -1 - N_pad,40):
                    for col_cent1 in range(N_pad,N_col -1 - N_pad,40):
                        img_temp = img1[(row_cent1-N_pad): (row_cent1+N_pad+1), (col_cent1-N_pad): (col_cent1+N_pad+1) ]  
                        if (np.sum([img_temp ==200] ) == 0)&(np.sum([img_temp > 1] ) >= 0.25*N_pixel):
                            for slice_id2 in search_list:
                                time_diff_list = [-5,0,5,10]
                                for time_diff in time_diff_list:
                                    T_id2 = T_id1 - time_diff
                                    if ((T_id2>0)&(T_id2<16)):
                                        img2,N_row,N_col = read_slice(input_file,slice_size,slice_id2,T_id2,H_id)
                                        match_result = cv2_based(img2,img_temp)
                                        if len(match_result[0])==1:
                                            search_list.remove(slice_id2)
                                            row_cent2 = match_result[0][0]+ N_pad
                                            col_cent2 = match_result[1][0]+ N_pad
                                            match_all.append( (slice_id1, row_cent1, col_cent1,T_id1, slice_id2 , row_cent2, col_cent2,T_id2) )
                    
    
    match_all_pd = pd.DataFrame(match_all,columns = ['slice_id1','row_id1','col_id1','T_id1','slice_id2','row_id2','col_id2','T_id2'])
    
    
    pd_add = pd.DataFrame(np.arange(1,N_slice+1), columns = ['slice_id1'])
    pd_add['slice_id2'] = pd_add['slice_id1']
    pd_add['row_id1'] = 10
    pd_add['row_id2'] = 10
    pd_add['T_id1'] = 10
    pd_add['col_id1'] = 10
    pd_add['col_id2'] = 10
    pd_add['T_id2'] = 10
    match_all_pd = pd.concat([match_all_pd,pd_add])
    match_all_pd.index = np.arange(len(match_all_pd))
    
    match_all_pd['row_diff'] = match_all_pd['row_id2'] - match_all_pd['row_id1']
    match_all_pd['col_diff'] = match_all_pd['col_id2'] - match_all_pd['col_id1']
    match_all_pd['time_diff'] = match_all_pd['T_id2'] - match_all_pd['T_id1']
    
    match_all_pd = match_all_pd.sort_values(by = ['slice_id1','slice_id2'])    
    
    #%%
    match_all_pd.index = np.arange(len(match_all_pd))
    
    for ind,value in match_all_pd.iterrows():
        if (value.slice_id2<value.slice_id1):
            temp = [value.T_id1,value.col_id1,value.row_id1,value.slice_id1]
            match_all_pd.loc[ind,['T_id1','col_id1','row_id1','slice_id1']] = match_all_pd.loc[ind,['T_id2','col_id2','row_id2','slice_id2']].values
            match_all_pd.loc[ind,['T_id2','col_id2','row_id2','slice_id2']] = temp
            match_all_pd.loc[ind,['row_diff',  'col_diff',  'time_diff'  ]] = - match_all_pd.loc[ind,['row_diff',  'col_diff',  'time_diff'  ]]
            
    match_all_pd = match_all_pd.sort_values(by = ['slice_id1','slice_id2'])
    match_all_pd.index = np.arange(len(match_all_pd))  
    match_all_pd.to_csv(data_folder + set_name + '_matches_time.csv', index = False)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    match_time_pd = pd.read_csv(data_folder  +set_name+'_matches_time.csv')
    if (set_name=='train'):
        match_all_pd = match_all_pd[~((match_all_pd.slice_id1==163)&(match_all_pd.slice_id2==165))]  
    
    N_match = len(match_all_pd)
    end_ind = 0
    slice_ind = 0
    Pano = []
    Pano_set_org = Set([1])
    for ind,value in match_all_pd.iterrows():
        Pano_set_new = Set(range(value.slice_id1,value.slice_id2+1))
        
        if len(Pano_set_org&Pano_set_new) >0:
            Pano_set_org = Pano_set_org|Pano_set_new
        else:
            Pano.append(sorted(Pano_set_org))
            Pano_set_org = Pano_set_new
    Pano.append(sorted(Pano_set_org))   
    
    Pano_time_list = []
    for i,pano in enumerate(Pano):
        Pano_time_list.append((i+1,pano[0],pano[-1]) )
        
    Pano_time_pd = pd.DataFrame(Pano_time_list, columns=['Sam_id','slice_sta','slice_end'])
    Pano_time_pd.to_csv(data_folder + set_name+'_panotime_index.csv',index = False)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    slice_size = pd.read_csv(data_folder + set_name+'_slice_size.csv')
    input_file = data_folder +set_name +'_slice_data'
    
#    match_all_pd = pd.read_csv(data_folder +set_name+'_matches_time.csv' )
    if (set_name =='train'):
        match_all_pd = match_all_pd[~((match_all_pd.slice_id1==163)&(match_all_pd.slice_id2==165))]  
        
    match_all_pd = match_all_pd.sort_values(by = ['slice_id1','slice_id2'])
    match_all_pd.index = np.arange(len(match_all_pd))
    
    #%% match_time_pd
    MATCH = pd.DataFrame()
    for ind,value in Pano_time_pd.iterrows():
        sam_id = value.Sam_id
        print(sam_id)
        slice_sta = value.slice_sta
        slice_end = value.slice_end
        slice_range = range(slice_sta,slice_end+1)
        slice_len = len(slice_range)
        slices = match_all_pd[ (match_all_pd.slice_id1.isin(slice_range))&(match_all_pd.slice_id2.isin(slice_range)) ]  
        path_mat = np.zeros([slice_len,slice_len]).astype(np.bool)    
        for ind2,value2 in slices.iterrows():
            path_mat[value2.slice_id1-slice_sta, value2.slice_id2-slice_sta] = True
        G = nx.from_numpy_matrix(path_mat)
        row_rel = []
        col_rel = []
        tim_rel = []
        for ind2 in range(slice_len):
            edge_path = nx.shortest_path(G, 0, ind2)
            row_shift_all = 0
            col_shift_all = 0
            tim_shift_all = 0
            for edge_ind in range(len(edge_path)-1):
                small_one = min(edge_path[edge_ind],edge_path[edge_ind+1])
                big_one = max(edge_path[edge_ind],edge_path[edge_ind+1])
                path_line = slices[(slices.slice_id1 == small_one  + slice_sta) & (slices.slice_id2 == big_one+slice_sta)]   
    
                if (edge_path[edge_ind+1]> edge_path[edge_ind]):         
                    row_shift = path_line.row_id2.values[0] - path_line.row_id1.values[0] 
                    col_shift = path_line.col_id2.values[0] - path_line.col_id1.values[0] 
                    tim_shift = path_line.T_id2.values[0] - path_line.T_id1.values[0] 
                else:
                    row_shift = - (path_line.row_id2.values[0] - path_line.row_id1.values[0] ) 
                    col_shift = - (path_line.col_id2.values[0] - path_line.col_id1.values[0] ) 
                    tim_shift = - (path_line.T_id2.values[0] - path_line.T_id1.values[0]  )
                    
                row_shift_all = row_shift_all + row_shift
                col_shift_all = col_shift_all + col_shift        
                tim_shift_all = tim_shift_all + tim_shift  
    
            row_rel.append(-row_shift_all)
            col_rel.append(-col_shift_all)
            tim_rel.append(-tim_shift_all)
        
        row_shift = np.asarray(row_rel)
        col_shift = np.asarray(col_rel)
        t_shift = np.asarray(tim_rel)
        
    
        row_list = []

        col_list = []
        for ind2,slice_ind in enumerate(slice_range):
            data_mat,row,col = read_slice(input_file,slice_size,slice_ind,1,1)
            row_list.append(row)
            col_list.append(col)
     
        
        ROW_min = np.min( row_shift)
        COL_min = np.min( col_shift)    
        
        row_shift = row_shift - ROW_min
        col_shift = col_shift - COL_min
        
        ROW_max = np.max(np.asarray(row_list) + row_shift)
        COL_max = np.max(np.asarray(col_list) + col_shift)       
        
        ROW_size = ROW_max - ROW_min 
        COL_size = COL_max - COL_min 
        
        MATCH_IND = pd.DataFrame( 
         {'SLI_ID': slice_range,
          'ROW_ID': row_shift,
          'COL_ID': col_shift,
          'TIM_ID' :t_shift
            })
        MATCH_IND['SAM_ID'] = sam_id
        MATCH_IND['ROW_MAX'] =  ROW_max
        MATCH_IND['COL_MAX'] =  COL_max
        MATCH_IND['TIM_MAX'] =  np.max(t_shift)+15
        
        MATCH = pd.concat([MATCH,MATCH_IND])
    
    MATCH = MATCH.sort_values(by = ['SAM_ID','SLI_ID'], ascending = [1,1])
    MATCH.index = np.arange(len(MATCH))
    
    MATCH.to_csv(data_folder + set_name+ '_TIME_MATCH.csv', index = False)
    
    
    output_file = data_folder + set_name +'_sample_data'
    f = open(output_file, "w")
    f.close()
    sample_stat =[]
    size_all = 0
    
    slice_stat_pd = pd.read_csv(data_folder + set_name+'_slice_size.csv')
    input_file = data_folder +set_name+'_slice_data'
        
    for sam_id in range(1,1+MATCH.SAM_ID.max()):   
        SAMS = MATCH[MATCH.SAM_ID == sam_id]
        print(SAMS)
        TIME_MAX = SAMS.TIM_MAX.max() 
        ROW_size = SAMS.ROW_MAX.max() 
        COL_size = SAMS.COL_MAX.max() 
        for t_id in range(1,1 + TIME_MAX):
            for H_id in range(1,5):
                TH_ind = (t_id-1)*4 + (H_id - 1)      
                view_all = (200*np.ones([ROW_size, COL_size])).astype(np.ubyte)

                for ind2,value in SAMS.iterrows():
                    slice_id = value.SLI_ID
                    row_id = value.ROW_ID
                    col_id = value.COL_ID
                    tim_id = value.TIM_ID
                    T_id = t_id - tim_id 
    
                    if ((T_id>=1)&(T_id<=15)):                  
                       data_mat,row,col = read_slice(input_file,slice_size,slice_id,T_id,H_id)
                       useful_place = np.where(data_mat!=200)
                       view_all[useful_place[0] + row_id, useful_place[1] + col_id] = data_mat[useful_place]
        
                if TH_ind ==0:  
                    sample_stat.append((sam_id, ROW_size,COL_size,TIME_MAX,size_all))
                    size_all = size_all + 4*TIME_MAX*ROW_size*COL_size
                f = open(output_file, "a")
                f.write(view_all.tobytes())  
                f.close()      
    #    
        
    sample_stat_pd = pd.DataFrame( sample_stat, columns = ['sample_id','N_row','N_col',' N_time','start_pos'])
    sample_stat_pd.to_csv(data_folder + set_name + '_sample_size.csv',index = False)      

time2 = time.time()
print('total elapse time:'+ str(time2- time1)) 