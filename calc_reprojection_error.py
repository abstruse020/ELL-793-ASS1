# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:38:44 2022

@author: bokar
"""

# import cv2 as cv
import numpy as np

# def reprojection_error_wrong():
#     mean_error = 0
#     for i in range(len(objpoints)):
#         imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#         error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#         mean_error += error
        
#     return mean_error
#%% Reprojection Error calculation

def reprojection_error(data, projection_mat):
    mean_error = 0
    pts_3d = np.ones((data.shape[0],4))
    # print(pts_3d.shape)
    pts_3d[:,:3] = data[:,:3]
    # print(pts_3d.shape)
    pts_2d = np.ones((data.shape[0],3))
    pts_2d[:,:2] = data[:,3:]
    for i in range(data.shape[0]):
        # print('Shape of 3d points:',pts_3d.shape)
        # print('Projection matrix shape:',projection_mat.shape)
        img_pt = np.dot(projection_mat, pts_3d[i])
        print('Image points calculated:', img_pt)
        img_pt =  img_pt/img_pt[-1]
        print('Normalized image point calculated:', img_pt)
        print('Original Image Points:', pts_2d[i])
        error = np.linalg.norm(img_pt - pts_2d[i])/data.shape[0]
        print('Error is:', error)
        mean_error+= error
        print()
    
    return mean_error

#%% Run this to test the above code

# data = np.random.rand(6,5)
# projection_mat = np.random.rand(3,4)

# reprojection_error(data, projection_mat)