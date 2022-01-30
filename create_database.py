# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:51:54 2022

@author: bokar
"""

import numpy as np
import cv2 as cv
import glob

#%% Creating Database
def get2d_points(image):
    '''
    Takes the image path and returns the 2d points for each corner
    
    Parameters
    ----------
    image : string
        Path of image to read and get 2d points

    Returns
    -------
    imgpoints : Numpy array (points_shape(7*6) X 1 X 2)
        Returns the 2d points of chess board

    '''
    print('Image path:',image)
    
    points_shape = (6,6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    img = cv.imread(image)
    #print(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    corners_present, corners = cv.findChessboardCorners(img, points_shape)
    print('Corners present:', corners_present)
    if corners_present:
        #print('Corners:', corners)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
         
        # Uncomment to see recognized corners
        cv.drawChessboardCorners(img, points_shape, corners2, corners_present)
        cv.imshow('img', img)
        cv.waitKey()
    return corners

#%% Running for all images in folder
images = glob.glob('D:\IITD\ELL793-CV\Assignment1\*.png')

all_pts_2d = []
for image in images:
    print('For image:',image)
    pts_2d = get2d_points(image)
    print('Shape of 2D points:', pts_2d.shape)
    all_pts_2d.append(pts_2d)
print(all_pts_2d[0][0])
print(all_pts_2d[0][1])
print(all_pts_2d[0][2])
print(all_pts_2d[0][3])
print(all_pts_2d[0][4])
print(all_pts_2d[0][5])
print(all_pts_2d[0][6])
print(all_pts_2d[0][7])
print(all_pts_2d[0][8])


#%%

