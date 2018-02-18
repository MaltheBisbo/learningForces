import numpy as np
import matplotlib.pyplot as plt
import math

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
    return angle, arg

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
    A = np.linalg.norm(RijVec/Rij - RikVec/Rik)
    B = np.linalg.norm(RijVec/Rij + RikVec/Rik)
    D = A/B
    RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
    RikMat = np.dot(RikVec[:,np.newaxis], RikVec[:,np.newaxis].T)

    a = RijVec/Rij - RikVec/Rik
    b = RijVec/Rij + RikVec/Rik
    
    a_grad_j = np.sum((-1/Rij**3 * RijMat + 1/Rij * np.identity(3)), axis=1)
    b_grad_j = np.sum((1/Rij**3 * RijMat - 1/Rij * np.identity(3)), axis=1)

    a_sum_j = np.sum(a*a_grad_j, axis=1)
    b_sum_j = np.sum(b*b_grad_j, axis=1)
    
    grad_j = 2/(1+D**2) * (1/(A*B) * a_sum_j - A/(B**3) * b_sum_j)

    


L = 2
d = 1
dim = 3
x = np.array([0.2*L, 0.7*L, d/2, 0.3*L, 0.2*L, d/2, 0.7*L, 0.9*L, 3*d/2])
positions = x.reshape((-1,dim))
ri = positions[0]
rj = positions[1]
rk = positions[2]
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
