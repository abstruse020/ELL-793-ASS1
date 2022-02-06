# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:24:19 2022

@author: bokar
"""

import numpy as np
from get_projection_mat import normalize_pts

points = np.random.rand(20,5)
# points[0,3:] = [1,2]
# points[1,3:] = [2,1]
print('points:\n', points)

n_pts, T, U = normalize_pts(points)

print('normalized points:\n',n_pts)

ans1 = 0
centr1 = 0
for i in range(points.shape[0]):
    ans1 += np.sqrt(np.sum(np.square(n_pts[i,:3])))
    centr1 += np.sum(n_pts[i,:3])

print('For 3D its should be root3:',ans1/points.shape[0], np.square(ans1/points.shape[0]))
print('Centroid should be 0 :', centr1)

ans2 = 0
centr2 = 0
for i in range(points.shape[0]):
    ans2 += np.sqrt(np.sum(np.square(n_pts[i,3:])))
    centr2 += np.sum(n_pts[i,3:])

print('For 2D its should be root2:',ans2/points.shape[0], np.square(ans2/points.shape[0]))
print('Centroid should be 0 :', centr2)

# ans1 = np.sqrt(np.sum(np.square(n_pts[0,:3])))
# ans2 = np.sqrt(np.sum(np.square(n_pts[1,:3])))
# print(ans1 + ans2)