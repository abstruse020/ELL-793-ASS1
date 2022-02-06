# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 01:16:03 2022

@author: bokar
"""

import numpy as np

from get_projection_mat import get_projection_matrix
from matrix_to_parameters import mat_to_parameters
from calc_reprojection_error import reprojection_error
from plot_it import plot_points_on_image


# Dataset from image

# data = [[0,	0,	0,	1792,	876],
        # [0,	4,	2,	1217,	731],
        # [0,	2,	3,	1547,	435],
        # [0,	6,	3,	780,    610],
        # [1,	0,	1,	1923,	769],
        # [4,	0,	0,	2335,	1139],
        # [2,	0,	3,	2102,	452],
        # [2,	3,	0,	1634,	1185],
        # [1,	5,	0,	1143,	1251],
        # [5,	2,	0,	2271,	1366]]


# data = [[0,	0,	0,	1792,	876],
#         [5,	2,	0,	2271,	1366],
#         [0,	4,	2,	1217,	731],
#         [1,	5,	0,	1143,	1251],
#         [0,	2,	3,	1547,	435],
#         [2,	0,	3,	2102,	452],
#         [0,	6,	3,	780,    610],
#         [1,	0,	1,	1923,	769],
#         [4,	0,	0,	2335,	1139],
#         [2,	3,	0,	1634,	1185]]


# Description
# Reprojection Error for Train Set: 0.281811199860872 for [0:6]
# Reprojection Error for Test Set: 7.611439207240161 without worst point
# Reprojection Error for Test Set: 15.089786761561303 with worst point

data = [[0,	0,	0,	1792,	876],
        [1,	5,	0,	1143,	1251],
        [2,	0,	3,	2102,	452],
        [0,	6,	3,	780,    610],
        [1,	0,	1,	1923,	769],
        [4,	0,	0,	2335,	1139],
        [2,	3,	0,	1634,	1185], # Not very good e^2:369
        #[5,	2,	0,	2271,	1366], #Not very good e^2:1825
        [0,	4,	2,	1217,	731],  #Not very good e^2:109
        [0,	2,	3,	1547,	435],  #Not very good e^2:64
        ]
data = np.array(data, dtype=(np.double))

# Main code to perform camera calibration

image = '.\chess_board_img_3d.jpg'

print('--------Projection Matrix---------')
projection_matrix = get_projection_matrix(np.array(data[:6], copy= True))
# print('Projection Matrix:\n', projection_matrix)
print()

print('--------Reprojection error--------')
train_rep_error, pred_xy = reprojection_error(data[:6], projection_matrix)
print('Reprojection Error for Train Set:',train_rep_error,'\n')

test_rep_error, pred_xy = reprojection_error(data[:], projection_matrix)
print('Reprojection Error for Test Set:',test_rep_error,'\n')
plot_points_on_image(np.array(pred_xy), data, image)


print('--------Camera values-------------')
print('intrinsic and extrensic\n')
values = mat_to_parameters(projection_matrix)






