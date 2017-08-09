# -*- coding: utf-8 -*-
"""
Transform the data type from ascii to ubyte format (8 bits unsigned binary) 
and save to new files, which would reduce the data size to 1/3, and would 
save the data transforming time when read by the python

@author: Marmot
"""

import numpy as np
import time
from itertools import islice
import pandas as pd
#

data_folder = '../../data/'
set_list = ['train','testA','testB']
size_list = [10000,2000,2000]
time1= time.time()

for set_name,set_size in zip(set_list,size_list):
    output_file = data_folder + set_name +  '_ubyte.txt'
    f = open(output_file, "w")
    f.close()
    Img_ind = 0
    input_file = data_folder + set_name +'.txt'
    with open(input_file) as f:
        for content in f:
            Img_ind = Img_ind +1 
            print('transforming ' + set_name  + ': '  + str(Img_ind).zfill(5))
            line = content.split(',')
            title = line[0] + '    '+line[1]        
            data_write = np.asarray(line[2].strip().split(' ')).astype(np.ubyte)
            data_write = (data_write + 1).astype(np.ubyte)          
            if data_write.max()>255:
                print('too large')
            if data_write.min()<0:
                print('too small')                
            f = open(output_file, "a")
            f.write(data_write.tobytes())  
            f.close()
time2 = time.time()
print('total elapse time:'+ str(time2- time1)) 

#%% generate train label list
value_list =[]
set_name = 'train'
input_file = data_folder + set_name +'.txt'
with open(input_file) as f:
    for content in f:
        line = content.split(',')
        value_list.append(float(line[1]))
value_list = pd.DataFrame(value_list, columns=['value'])
value_list.to_csv(data_folder + 'train_label.csv',index = False,header = False)

