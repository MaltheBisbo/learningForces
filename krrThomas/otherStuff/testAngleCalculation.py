import numpy as np
import matplotlib.pyplot as plt
import math

import pdb

def angle1(v1,v2):
    """
    Returns angle with convention [0,pi]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    arg = np.dot(v1,v2)/(norm1*norm2)
    # This is added to correct for numerical errors
    if arg < -1:
        arg = -1.
    elif arg > 1:
        arg = 1.
    angle = np.arccos(arg)
    return angle

def angle2(ri, rj, rk):
    RijVec = rj - ri
    RikVec = rk - ri
    Rij = np.linalg.norm(RijVec)
    Rik = np.linalg.norm(RikVec)
    A = np.linalg.norm(RijVec/Rij - RikVec/Rik)
    B = np.linalg.norm(RijVec/Rij + RikVec/Rik)
    angle = 2*np.arctan2(A, B)
    return angle

def angle2_grad(ri, rj, rk):
    RijVec = rj - ri
    RikVec = rk - ri
    Rij = np.linalg.norm(RijVec)
    Rik = np.linalg.norm(RikVec)
    
    a = RijVec/Rij - RikVec/Rik
    b = RijVec/Rij + RikVec/Rik
    A = np.linalg.norm(a)
    B = np.linalg.norm(b)
    D = A/B

    RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
    RikMat = np.dot(RikVec[:,np.newaxis], RikVec[:,np.newaxis].T)

    a_grad_j = -1/Rij**3 * RijMat + 1/Rij * np.identity(3)
    b_grad_j = a_grad_j

    a_sum_j = np.sum(a*a_grad_j, axis=1)
    b_sum_j = np.sum(b*b_grad_j, axis=1)
    
    grad_j = 2/(1+D**2) * (1/(A*B) * a_sum_j - A/(B**3) * b_sum_j)



    a_grad_k = 1/Rik**3 * RikMat - 1/Rik * np.identity(3)
    b_grad_k = -a_grad_k

    a_sum_k = np.sum(a*a_grad_k, axis=1)
    b_sum_k = np.sum(b*b_grad_k, axis=1)
    
    grad_k = 2/(1+D**2) * (1/(A*B) * a_sum_k - A/(B**3) * b_sum_k)


    a_grad_i = -(a_grad_j + a_grad_k)
    b_grad_i = -(b_grad_j + b_grad_k)

    a_sum_i = np.sum(a*a_grad_i, axis=1)
    b_sum_i = np.sum(b*b_grad_i, axis=1)
    
    grad_i = 2/(1+D**2) * (1/(A*B) * a_sum_i - A/(B**3) * b_sum_i)
    
    return grad_i, grad_j, grad_k

Natoms = 3
L = 2
d = 1
dim = 3
theta = 1

dx = 0.001
Npoints = 199
angle_old = np.zeros(Npoints)
angle = np.zeros(Npoints)
angle_grad = np.zeros((Npoints, dim))
angle_grad_num = np.zeros((Npoints, dim))

theta_fixed = 0.2
theta_array = np.linspace(-2*np.pi, 2*np.pi, Npoints)
for i, theta in enumerate(theta_array):
    x = np.array([0, 0, 0,
                  np.cos(theta), np.sin(theta), 0,
                  np.cos(theta_fixed), np.sin(theta_fixed), 0])
                  
    pos = x.reshape((-1,dim))
    ri = pos[0]
    rj = pos[1]
    rk = pos[2]
    RijVec = rj - ri
    RikVec = rk - ri

    angle_old[i] = angle1(RijVec, RikVec)
    angle[i] = angle2(ri, rj, rk)
    angle_grad[i], _, _ = angle2_grad(ri, rj, rk)
    
    for j in range(dim):
        x_pertub = np.zeros(Natoms*dim)
        x_pertub[j] = dx

        pos_down = np.reshape(x - x_pertub/2, (-1, dim))
        pos_up = np.reshape(x + x_pertub/2, (-1, dim))

        angle_down = angle2(pos_down[0], pos_down[1], pos_down[2])
        angle_up = angle2(pos_up[0], pos_up[1], pos_up[2])
        angle_grad_num[i,j] = (angle_up - angle_down)/dx
        
plt.figure(1)
plt.plot(theta_array, angle_old)
plt.plot(theta_array, angle, linestyle=':')

plt.figure(2)
plt.plot(theta_array, angle_grad_num)
plt.plot(theta_array, angle_grad, linestyle=':')
plt.show()





















"""
RijVec = rj - ri
RikVec = rk - ri
Rij = np.linalg.norm(RijVec)
Rik = np.linalg.norm(RikVec)

print('ri:', ri)
print('rj:', rj)
print('rk:', rk)

print('RijVec:', RijVec)
print('RikVec:', RikVec)

RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
a = RijVec/Rij - RikVec/Rik
print('RijMat:\n', RijMat)
print('a:', a)
print(a*RijMat)
print(np.sum(a*RijMat, axis=1))
# Angle using function angle1()
print('angle using angle1():', angle1(RijVec, RikVec))

# Angle using function angle2()
print('angle using angle2():', angle2(ri, rj, rk))
"""
