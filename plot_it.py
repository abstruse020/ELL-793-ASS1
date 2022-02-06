# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 23:21:26 2022

@author: bokar
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_points_on_image(points, data, image):
    img = mpimg.imread(image)
    plt.scatter(points[:,0],points[:,1], s=5, alpha=0.8, label = 'Predicted Points')
    plt.scatter(data[:,3],data[:,4], s=5, alpha=0.5, label = 'Actual Points', c= 'gray')
    # plt.plot(img)
    plt.legend()
    plt.imshow(img)
    plt.savefig('pred.png', dpi = 300)
    # plt.imsave('pred.png',img)
    # plt.show()
    # plt.plot(points)

#%%
# image = 'd:\IITD\ELL793-CV\Assignment1\chess_board_img_3d.jpg'
# points = [[500, 500],
#           [1000,1000],
#           [1000,1500]]
# points = np.array(points)
# data = np.random.rand(3,5)

# plot_points_on_image(points, data, image)
