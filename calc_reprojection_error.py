# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:38:44 2022

@author: bokar
"""

import cv2 as cv
import numpy as np

#%% reprojection error function

def reprojection_error():
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        
    return mean_error
