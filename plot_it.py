# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 23:21:26 2022

@author: bokar
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_points_on_image(points, image):
    img = mpimg.imread(image)
    plt.scatter(points[:,0],points[:,1], s=5, alpha=0.5)
    #plt.plot(img)
    plt.imshow(img)
    # plt.imsave('pred.png',img)
    # plt.show()
    # plt.plot(points)

#%%
# image = 'd:\IITD\ELL793-CV\Assignment1\chess_board_img_3d.jpg'
# points = [[500, 500],
#           [1000,1000],
#           [1000,1500]]
# points = np.array(points)

# plot_points_on_image(points, image)