# Python deomo code to readin and plot the radar signal data

# create by Fernando from Team Marmot

import numpy as np

import linecache

import matplotlib.pyplot as plt

#%%

def read_plot(Img_ind,TH_ind,file_name):

    Img_size = 101

    line = (linecache.getline(file_name, Img_ind)).split(',')

    image_name = line[0] + '    '+line[1]

    print(image_name)

    img_line = np.asarray(line[2].strip().split(' ')[TH_ind*Img_size**2:(TH_ind+1)*Img_size**2]).astype(np.ubyte)

    img_mat = img_line.reshape([Img_size,Img_size])    

    plt.imshow(img_mat )

    title = 'image_id:' + str(Img_ind).zfill(5) + '  time_id:' + str(T_ind) + '  height_ind:' + str(H_ind)

    plt.title(title)

    return 0

#%%
if __name__ == '__main__':    

    T_ind = 15   # time index: 1 - 15

    H_ind = 4    # height index:  1 - 4 (0.5km - 3.5 km)

    Img_ind = 1  # image index: 1-10000 in train, 1- 2000 in testA

    TH_ind = (T_ind-1)*4 + (H_ind - 1)

    file_name = '../data/testA.txt'    

    

    read_plot(Img_ind,TH_ind,file_name)

