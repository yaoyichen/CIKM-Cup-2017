# -*- coding: utf-8 -*-
"""
Spatial template matching of sub-images

@author: Marmot
"""
import sys
sys.path.append('../TOOLS')
from CIKM_TOOLS import *

L_img = 101   # size of image 
N_pad = 3     # Pad size of matching template
N_block = 5   # number of blocks along each side of image
N_cor = N_block**2
time1 = time.time()        
data_folder = '../../data/'


set_list = ['train','testA','testB']
size_list = [10000,2000,2000]
time1= time.time()

for set_name,N_pic in zip(set_list,size_list):
    input_file = data_folder + set_name + '_ubyte.txt'
    
    block_ele = np.asarray([N_pad,(L_img-N_pad)/4,(L_img-N_pad)/2,(L_img-N_pad)*3/4,(L_img - N_pad - 1)])

    cor_row_center = np.tile(block_ele,[N_block,1]).reshape([1,N_cor], order='F')[0]
    cor_col_center = np.tile(block_ele,N_block)


    print (cor_row_center)
    print (cor_col_center)

    sys.exit(2)
    
    match_all = []
    
    #==============================================================================
    for pic_id1 in range(1,N_pic+1):  
        print('matching ' + set_name +': ' +str(pic_id1).zfill(5))
        N_CHANGE = 0
        for T_id in range(1,16,3):
            for H_id in range(2,5):
                FAIL_CORNER = 0
                data_mat1 = read_data(input_file,pic_id1,T_id,H_id)       
                search_list = range( max((pic_id1-10),1),pic_id1)+  range(pic_id1+1, min((pic_id1 + 16),N_pic + 1 ) ) 
                
                for cor_ind in range(0,N_cor):
                    row_cent1 = cor_row_center[cor_ind]
                    col_cent1 = cor_col_center[cor_ind]
                    img_corner = data_mat1[(row_cent1-N_pad): (row_cent1+N_pad+1), (col_cent1-N_pad): (col_cent1+N_pad+1) ]
                    if ((len(np.unique(img_corner))) >2)&(np.sum(img_corner ==1)< 0.8*(N_pad*2+1)**2)  :       
                    
                        for pic_id2 in search_list:    
                            data_mat2 = read_data(input_file,pic_id2,T_id,H_id)
                            match_result = cv2_based(data_mat2,img_corner)
                            if len(match_result[0]) ==1:
                                row_cent2 = match_result[0][0]+ N_pad
                                col_cent2 = match_result[1][0]+ N_pad
                                N_LEF = min( row_cent1 , row_cent2)
                                N_TOP = min( col_cent1,  col_cent2 )                        
                                N_RIG = min( L_img-1-row_cent1 , L_img-1-row_cent2)
                                N_BOT = min( L_img-1-col_cent1 , L_img-1-col_cent2)                       
                                IMG_CHECK1 = data_mat1[(row_cent1-N_LEF): (row_cent1+N_RIG+1), (col_cent1-N_TOP): (col_cent1+N_BOT+1) ]
                                IMG_CHECK2 = data_mat2[(row_cent2-N_LEF): (row_cent2+N_RIG+1), (col_cent2-N_TOP): (col_cent2+N_BOT+1) ]
                                if np.array_equal(IMG_CHECK1,IMG_CHECK2) :
                                    check_row_N = IMG_CHECK1.shape[0]
                                    check_col_N = IMG_CHECK1.shape[1] 
                                    if (check_col_N*check_row_N>=25):
                                        match_all.append( (pic_id1, row_cent1, col_cent1, pic_id2 , row_cent2, col_cent2) )
                                        search_list.remove(pic_id2)
                    else:
                        FAIL_CORNER = FAIL_CORNER +1 
        
                N_CHANGE = N_CHANGE + 1
                
                #%% break if less than 1 useless corners, or have detected more than 10 images from 60
                if(FAIL_CORNER <= 1):
                    break
                     
    
    match_all_pd = pd.DataFrame(match_all,columns = ['pic_id1','row_id1','col_id1','pic_id2','row_id2','col_id2'])
    pd_add = pd.DataFrame(np.arange(1,N_pic+1), columns = ['pic_id1'])
    pd_add['pic_id2'] = pd_add['pic_id1']
    pd_add['row_id1'] = 0
    pd_add['row_id2'] = 0
    pd_add['col_id1'] = 0
    pd_add['col_id2'] = 0
    match_all_pd = pd.concat([match_all_pd,pd_add])
    match_all_pd.index = np.arange(len(match_all_pd))
       
    for ind,value in match_all_pd.iterrows():
        if value.pic_id1 <= value.pic_id2:
            continue
        else:
            temp = value.pic_id2
            match_all_pd.loc[ind,'pic_id2'] = value.pic_id1
            match_all_pd.loc[ind,'pic_id1'] = temp       
            temp = value.row_id2
            match_all_pd.loc[ind,'row_id2'] = value.row_id1
            match_all_pd.loc[ind,'row_id1'] = temp  
            temp = value.col_id2
            match_all_pd.loc[ind,'col_id2'] = value.col_id1
            match_all_pd.loc[ind,'col_id1'] = temp
            
    
    match_all_pd['row_diff'] = match_all_pd['row_id2'] - match_all_pd['row_id1']
    match_all_pd['col_diff'] = match_all_pd['col_id2'] - match_all_pd['col_id1']
    match_all_pd = match_all_pd.sort_values(by = ['pic_id1','pic_id2'])
    match_all_pd = match_all_pd.drop_duplicates(subset = ['pic_id1','pic_id2','row_diff','col_diff'],keep = 'first')
    
    match_check = match_all_pd.groupby(by =['pic_id1','pic_id2','row_diff','col_diff']).count()
    if(len(match_check[match_check.col_id1>1])>0):
        print('error')
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    match_all_pd = match_all_pd.sort_values(by = ['pic_id1','pic_id2'],ascending = [1,1])
    match_all_pd = match_all_pd.drop_duplicates(subset = ['pic_id1','pic_id2','row_diff','col_diff'],keep = 'first')
    #%%                                                                      
      
    MATCH = pd.DataFrame()
    end_ind = 0
    slice_ind = 0
    while(end_ind<N_pic):
        slice_ind = slice_ind + 1
        success = 1
        start_ind = end_ind+1
        pad_ind = start_ind
        curr_ind = start_ind
        
        Pano_set_org = Set()
        while (1):
            match_all_pd_start = match_all_pd[match_all_pd.pic_id1 == curr_ind]
            Pano_set_new = Set(match_all_pd_start.pic_id2.values)
            Pano_set_org = Pano_set_org|Pano_set_new
            try:
                end_ind = max(list(Pano_set_org) )
            except:
                end_ind = start_ind
            
            if (curr_ind == end_ind):
                break
            curr_ind = curr_ind + 1       
        Pano_set_org = sorted(Pano_set_org)  
        
        #%%    
        match_all_pd_current = match_all_pd[match_all_pd.pic_id1.isin(Pano_set_org)]
        match_all_pd_current = match_all_pd_current[match_all_pd_current.pic_id2.isin(Pano_set_org)]
        
        #%%
        Pano_len = len(Pano_set_org)
        path_mat = np.zeros([Pano_len,Pano_len]).astype(np.bool)
        for ind in list(Pano_set_org):
            match_all_pd_start = match_all_pd_current[match_all_pd_current.pic_id1 == ind]
            for ind2,value2 in match_all_pd_start.iterrows():          
                path_mat[value2.pic_id1-start_ind, value2.pic_id2-start_ind] = True
                           
        #%%
        
        G = nx.from_numpy_matrix(path_mat)
        
        pic_list = []
        row_rel = []
        col_rel = []
        
        for ind in range(Pano_len):
            try:
                edge_path = nx.shortest_path(G, 0, ind)
            except:
                print(Pano_set_org[ind])
                continue
            row_shift_all = 0
            col_shift_all = 0
            for edge_ind in range(len(edge_path)-1):
                small_one = min(edge_path[edge_ind],edge_path[edge_ind+1])
                big_one = max(edge_path[edge_ind],edge_path[edge_ind+1])
                path_line = match_all_pd[(match_all_pd.pic_id1 == small_one  + pad_ind) & (match_all_pd.pic_id2 == big_one+pad_ind)]   
    
                
                if (edge_path[edge_ind+1]> edge_path[edge_ind]):         
                    row_shift = path_line.row_id2.values[0] - path_line.row_id1.values[0] 
                    col_shift = path_line.col_id2.values[0] - path_line.col_id1.values[0] 
                else:
                    row_shift = - (path_line.row_id2.values[0] - path_line.row_id1.values[0] ) 
                    col_shift = - (path_line.col_id2.values[0] - path_line.col_id1.values[0] ) 
    
                row_shift_all = row_shift_all + row_shift
                col_shift_all = col_shift_all + col_shift        
    
            pic_list.append(ind+ pad_ind)
            row_rel.append(-row_shift_all)
            col_rel.append(-col_shift_all)
            
    
    
        x = np.asarray(row_rel)
        y = np.asarray(col_rel)
    
        x_shift = x - np.min(x) + 50
        y_shift = y - np.min(y) + 50
        
        T_ind = 1
        H_ind = 4
        
        view_all = (200*np.ones([np.max(x_shift)+51, np.max(y_shift)+51])).astype(np.ubyte)
        for ind in range(len(pic_list )):
            pic_ind = pic_list[ind]
            img = read_data(input_file,pic_ind,T_ind,H_ind)#        
            img_check = view_all[x_shift[ind]-50:x_shift[ind]+51, y_shift[ind]-50:y_shift[ind]+51]
            if (np.sum((img_check == img) | (img_check == 200)) < 10201):
                print('fail number:' + str(pic_ind) )
                success = 0
            
            view_all[x_shift[ind]-50:x_shift[ind]+51, y_shift[ind]-50:y_shift[ind]+51] = img
    
        
        MATCH_IND = pd.DataFrame( 
         {'PIC_IND': pic_list,
          'ROW_IND': x_shift,
          'COL_IND': y_shift
            })
            
        MATCH_IND['SLICE_IND'] = slice_ind    
        MATCH_IND['SUCCESS'] = success
    
        MATCH = pd.concat([MATCH, MATCH_IND ], axis = 0)
    MATCH = MATCH[['SLICE_IND','PIC_IND','ROW_IND','COL_IND']]    
    MATCH.to_csv(data_folder + set_name + '_MATCH.csv',index = False)   
    #%%
    
    
    output_file = data_folder + set_name + '_slice_data'
    
    f = open(output_file, "w")
    f.close()
      
    size_all = 0
    slice_stat = []
    N_slice = MATCH['SLICE_IND'].max()
    for slice_ind in range(1, 1+N_slice):
        print('slice_ind:' + str(slice_ind))
        MATCH_slice = MATCH[MATCH.SLICE_IND == slice_ind]
        ROWS = MATCH_slice['ROW_IND'].max() + 51
        COLS = MATCH_slice['COL_IND'].max() + 51
        for T_ind in range(1,16):
            for H_ind in range(1,5):
            
                TH_ind = (T_ind-1)*4 + (H_ind - 1)            
                view_all = (200*np.ones([ROWS, COLS])).astype(np.ubyte)
                for ind2,value in MATCH_slice.iterrows():
                    pic_ind = value.PIC_IND
                    row_ind = value.ROW_IND
                    col_ind = value.COL_IND
                    img = read_data(input_file,pic_ind,T_ind,H_ind)#             
                    view_all[row_ind-50:row_ind+51, col_ind-50:col_ind+51] = img
    #            
                if TH_ind ==0:
                    slice_stat.append((slice_ind, view_all.shape[0],view_all.shape[1],size_all))
                    size_all = size_all + 60*view_all.shape[0]*view_all.shape[1]
                f = open(output_file, "a")
                f.write(view_all.tobytes())  
                f.close()  
    #               
    slice_stat_pd = pd.DataFrame( slice_stat, columns = ['slice_id','rows','cols','start_pos'])
    slice_stat_pd.to_csv(data_folder + set_name + '_slice_size.csv',index = False)

time2 = time.time()
print('total elapse time:'+ str(time2- time1)) 

