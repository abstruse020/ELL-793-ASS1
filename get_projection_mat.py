# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 17:45:58 2022

@author: harsh
"""
import numpy as np

#%% Find Projection matrix

def normalize_pts(data):
    
    p_2d = data[:,3:]
    p_3d = data[:,:3]
    
    print('2D points:', p_2d)
    print('3D points:', p_3d)
    
    mean_2d = np.mean(p_2d, 0)
    mean_3d = np.mean(p_3d, 0)
    std_2d = np.std(p_2d)
    std_3d = np.std(p_3d)
    
    T = np.array([
        [std_2d, 0, mean_2d[0]], 
        [0, std_2d, mean_2d[1]],
        [0, 0, 1]
        ])
    U = np.array([
        [std_3d, 0, 0, mean_3d[0]],
        [0, std_3d, 0, mean_3d[1]],
        [0, 0, std_3d, mean_3d[2]],
        [0, 0, 0, 1]])
    T = np.linalg.inv(T)
    U = np.linalg.inv(U)
    
    p_2d = np.dot( T, np.concatenate((p_2d.T, np.ones((1, p_2d.shape[0])))))
    p_3d = np.dot( U, np.concatenate((p_3d.T, np.ones((1, p_3d.shape[0])))))
    
    data[:, 3:] = p_2d[0:2,:].T
    data[:, :3] = p_3d[0:3,:].T
    
    return data, T, U 

def denormalize_pts(projection_matrix,T, U):
    proj = np.dot( np.dot( np.linalg.inv(T), projection_matrix), U)
    proj = proj/proj[-1,-1]
    return proj

def solve_homogeneous_eqn(A,b):
    '''

    Parameters
    ----------
    A : 12 X 12 numpy matrix
        P matrix on LHS of eqn
    b : 12 X 1 numpy matrix
        matrix of Zero on RHS of eqn
        
    eqn => A @ x = b
    Here b = zero vector
    x is the required 12 values of projection matrix
    
    Returns
    -------
    x = list of 12 params of projection matrix

    '''
    U, Sig, V = np.linalg.svd(A)
    
    x= V[-1,:] / V[-1,-1]
    
    return x.reshape(3,4)
    

def get_projection_matrix(data):
    '''
    
    Parameters
    ----------
    data : n X 5 list or np array
        n = number of input images
        5 = X,Y,Z and x,y points (i.e 3D and 2D points)

    Returns
    -------
    Projection matrix.
    
    '''
    debug = False
    n  = len(data)
    Pi = np.ones((n, 4))
    P  = np.zeros((2*n, 12))
    X  = np.zeros(n)
    Y  = np.zeros(n)
    
    # Normalizing the data
    data, T, U = normalize_pts(data)
    
    print('Pi matrix:') if debug  else None
    for i, row in enumerate(data):
        print(row) if debug else None
        Pi[i][:3] = row[:3]
        X[i] = row[3]
        Y[i] = row[4]
        print(Pi[i]) if debug else None
    
    print('--------------------------------') if debug else None
    print('making P matrix') if debug else None
    index = 0
    for i in range(0, 2*n ,2):
        print('for i:',i) if debug else None
        row1 = list(Pi[index]) + [0,0,0,0] + list(-1 * X[index] * Pi[index])
        row2 = [0,0,0,0] + list(Pi[index]) + list(-1 * Y[index] * Pi[index])
        # row1 = [Pi[index], np.zeros(4), -1 * X[index] * Pi[index]]
        # row2 = [np.zeros(4), Pi[index], -1 * Y[index] * Pi[index]]
        P[i] = row1
        P[i+1] = row2
        index += 1
        print(P[i]) if debug else None
        print(P[i+1]) if debug else None
    
    print('-------------------------------')
    print('shape of P', P.shape)
    b = np.zeros(2*n)
    
    # Calculating projection matrix using least square
    #projection_matrix = np.linalg.lstsq(P, b, rcond = None)[0]
    
    # Calculating projection matric using SVD
    projection_matrix = solve_homogeneous_eqn(P, b)
    
    # Denormalize the matrix
    projection_matrix = denormalize_pts(projection_matrix, T, U)
    
    print('shape of prjection matrix:\n',projection_matrix.shape)
    print('complete projection matrix:\n', projection_matrix)
    return projection_matrix

#%% Run above

data = np.random.rand(6,5)
print('data->\n',data)

get_projection_matrix(data)


