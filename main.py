# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 01:16:03 2022

@author: bokar
"""

import numpy as np
# import scipy.linalg
# import mpmath

from get_projection_mat import get_projection_matrix
from matrix_to_parameters import mat_to_parameters
from calc_reprojection_error import reprojection_error


# Dataset from image

# data = [[0,	0,	0,	1792,	876],
        # [0,	4,	2,	1217,	731],
        # [0,	2,	3,	1547,	435],
        # [0,	6,	3,	780,    610],
        # [1,	0,	1,	1964,	769],
        # [4,	0,	0,	2335,	1139],
        # [2,	0,	3,	2102,	452],
        # [2,	3,	0,	1632,	1187],
        # [5,	1,	0,	1141,	1241],
        # [2,	5,	0,	2261,	1361]]


data = [[0,	0,	0,	1792,	876],
        [2,	5,	0,	2261,	1361],
        [0,	4,	2,	1217,	731],
        [5,	1,	0,	1141,	1241],
        [0,	2,	3,	1547,	435],
        [2,	0,	3,	2102,	452],
        [0,	6,	3,	780,    610],
        [1,	0,	1,	1964,	769],
        [4,	0,	0,	2335,	1139],
        [2,	3,	0,	1632,	1187]]

data = np.array(data, dtype=(float))
print('Type of data:',type(data))
print('Complete Data:\n', data)

# Main code to perform camera calibration

projection_matrix = get_projection_matrix(data[:])
print('Projection Matrix:\n', projection_matrix)

train_rep_error = reprojection_error(data[:6], projection_matrix)
print('Reprojection Error for Train Set:',train_rep_error,'\n')

test_rep_error = reprojection_error(data[6:], projection_matrix)
print('Reprojection Error for Test Set:',test_rep_error,'\n')

values = mat_to_parameters(projection_matrix)






