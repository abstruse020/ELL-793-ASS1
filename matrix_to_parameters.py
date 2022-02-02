#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.linalg
import mpmath

# P = np.array([[3.53553e+2,  3.39645e+2, 2.77744e+2, -1.44946e+6],
#               [-1.03528e+2, 2.33212e+1, 4.59607e+2, -6.3252e+5],
#               [7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e+2]])

def mat_to_parameters(P):
    M = P[:,0:3]
    
    K,R = scipy.linalg.rq(M)
    
    T = np.diag(np.sign(np.diag(K)))
    
    if scipy.linalg.det(T) < 0:
        T[1,1] *= -1
    
    K = np.dot(K,T)
    R = np.dot(T,R)
    
    C = np.dot(scipy.linalg.pinv(-M),P[:,3])
    
    alpha=K[0][0]
    x0=K[0][2]
    y0=K[1][2]
    tan_theta=K[0][0]/(-K[0][1])
    theta=mpmath.atan(tan_theta)
    print("Alpha",alpha)
    print("the difference between the  centre of camera coordinate frame and the image center along the x axis",x0)
    print("the difference between the  centre of camera coordinate frame and the image center along the y axis",y0)
  
    sin_theta=mpmath.sin(theta)
    beta=K[1][1]*sin_theta
  
    T=-(np.dot(R,C))
    fx=K[0][0]
    fy=K[1][1]
    s=K[0][1]
    print("intrinsic camera matrix",K)
    print("Translation matrix ",T)
    print("Rotation matrix",R)
  
    print("camera focal length in the x axis in pixels",fx)
    print("camera focal length in the y axis in pixels",fy)
    print("skew parameter",s)
    print("The angle between the x-axis and y-axis in the image plane",theta)


   


# mat_to_parameters(P)


# In[ ]:




