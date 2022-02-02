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
    debug = False
    mean_error = 0
    
    pts_3d = np.ones((data.shape[0],4))
    pts_3d[:,:3] = data[:,:3]
    pts_2d = np.ones((data.shape[0],3))
    pts_2d[:,:2] = data[:,3:]
    pred_xy = []
    print('3D points:\n', pts_3d) if debug else None
    print('2D points:\n', pts_2d) if debug else None
    
    for i in range(data.shape[0]):
        print('Shape of 3d points:',pts_3d[i].shape) if debug else None
        print('Projection matrix shape:',projection_mat.shape) if debug else None
        img_pt = np.dot(projection_mat, pts_3d[i])
        print('Image points calculated:', img_pt) if debug else None
        img_pt =  img_pt/img_pt[-1]
        pred_xy.append(img_pt)
        print('Normalized image point calculated:', img_pt) if debug else None
        print('Original Image Points:', pts_2d[i]) if debug else None
        error = np.sum(np.square(img_pt[:2] - pts_2d[i,:2]))
        print('Error sq is:', error) if debug else None
        mean_error+= error/data.shape[0]
        print() if debug else None
    mean_error = np.sqrt(mean_error)
    return mean_error, img_pt

#%% Run this to test the above code

data = np.random.rand(6,5)
projection_mat = np.random.rand(3,4)

reprojection_error(data, projection_mat)
#%%

mat1 = [[1,2,1],
        [0,4,5]]
mat2 = [[3,4,5],
        [4,5,6]]
mat2 = np.array(mat2)
print(np.dot(mat1, mat2[0].T))

