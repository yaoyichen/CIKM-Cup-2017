# -*- coding: utf-8 -*-
"""

import libs and commonly used functions

@author: Marmot
"""

import random
import numpy as np
import time
from sets import Set
import pandas as pd
import os
import networkx as nx
import cv2
from cv2 import matchTemplate as cv2m
import networkx as nx
import matplotlib.animation as animation
from PIL import Image
from itertools import islice
import math
from sklearn import linear_model
from sklearn import preprocessing
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


def read_data(input_file,pic_ind,T_ind,H_ind):
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( ((pic_ind -1)*60 + TH_ind)*101*101, os.SEEK_SET)  # seek
    data = np.fromfile(  f, count = 101*101, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(101,101)
    return data_mat

def read_slice(input_file, size_file, slice_ind,T_ind,H_ind):
    _,row,col,pos = size_file[size_file.slice_id ==slice_ind].values[0]
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( pos + TH_ind*row*col , os.SEEK_SET)  # seek
    data = np.fromfile(  f, count = row*col, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(row,col)
    return data_mat,row,col

def read_sample(input_file,input_size, sample_ind,T_ind,H_ind):
    tt= input_size[input_size.testB_SAM_ID == sample_ind]
    pos = tt.start_pos.values[0]
    row = tt.N_row.values[0]
    col=  tt.N_col.values[0]
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( pos + TH_ind*row*col , os.SEEK_SET)  # seek
    data = np.fromfile(  f, count = row*col, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(row,col)
    return data_mat

def read_sample_trn(input_file,input_size,sample_ind,T_ind,H_ind):
    _,row,col,time,pos = input_size[input_size.sample_id == sample_ind].values[0]
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( pos + TH_ind*row*col , os.SEEK_SET)  # seek
    data = np.fromfile(  f, count = row*col, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(row,col)
    return data_mat

def read_sample_AB(input_file,input_size, sample_ind,T_ind,H_ind):
    tt= input_size[input_size.testB_SAM_ID == sample_ind]
    pos = tt.start_pos.values[0]
    row = tt.N_row.values[0]
    col=  tt.N_col.values[0]
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( pos + TH_ind*row*col , os.SEEK_SET)  # seek
    data = np.fromfile(  f, count = row*col, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(row,col)
    return data_mat

# All the 6 methods for comparison in a list
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']  
def cv2_based(field_array,match_array):
    M = cv2m(field_array,match_array,cv2.TM_SQDIFF_NORMED)
    return np.where(1e-6>=M)
	

def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 2
    thickness = 1
    print(len(kp1),len(kp2), len(matches) )
    if color:
        c = color
    for m in matches[0:20]:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        c = [255,255,255]
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))

        
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()
	
	
def match(desc1,desc2):
    """ For each descriptor in the first image, 
        select its match in the second image.
        input: desc1 (descriptors for the first image), 
        desc2 (same for second image). """
    
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.95
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0]),'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))
        
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    
    return matchscores

	
def match_twosided(desc1,desc2):
    """ Two-sided symmetric version of match(). """
    
    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    
    return matches_12
    

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    
    return np.concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
        matchscores (as output from 'match'), show_below (if images should be shown below). """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))
    
    # show image
    plt.imshow(im3,cmap=plt.cm.gray)
    
    # draw lines for matches
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.scatter([locs1[i][0],locs2[m][0]+cols1],[locs1[i][1],locs2[m][1]],c= 'r',s=30,marker='o')
            plt.plot([locs1[i][0],locs2[m][0]+cols1],[locs1[i][1],locs2[m][1]],'y')
    plt.axis('off')
    plt.grid(linestyle='solid',color = 'gray', linewidth = 1)
    


from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion



def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks
    
def lr_model(train_pic_sample,feature_list):

    Y = train_pic_sample['value'].values.ravel()
    X = train_pic_sample[feature_list].values
    
    scaler = preprocessing.MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X) 
    model = linear_model.LinearRegression()
    model.fit(X_scaled,  Y)
    y_true = Y
    y_pred = model.predict(X_scaled)
    y_pred[y_pred<0] = 1.0
    
    print('train mean:' + str(np.mean(y_true)) ) 
    print('train mean pred:' + str(np.mean(y_pred)) ) 
    print('train mse:' + str(np.std(y_true - y_pred) ) ) 