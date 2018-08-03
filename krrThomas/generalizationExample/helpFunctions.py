import numpy as np
from doubleLJ import doubleLJ_energy_ase as E_doubleLJ
import matplotlib.pyplot as plt

from ase import Atoms


def cartesian_coord(x1,x2):
    """
    Calculates the 2D cartesian coordinates from the transformed coordiantes.
    """
    d0 = 1.07
    theta1 = -np.pi/3
    theta2 = -2/3*np.pi
    pos0 = np.array([0,0])
    pos1 = np.array([-d0 + x1*np.cos(theta1), x1*np.sin(theta1)])
    pos2 = np.array([d0 + x2*np.cos(theta2), x2*np.sin(theta2)])

    pos = np.array([pos1, pos0, pos2])
    return pos

def structure(x1,x2):
    '''
    x1, x2 is the two new coordiantes
    d0 is the equilibrium two body distance
    '''
    d0 = 1.07
    x1 = x1*d0
    x2 = x2*d0
    theta1 = -np.pi/3
    theta2 = -2/3*np.pi
    pos0 = np.array([0,0,0])
    pos1 = np.array([-d0 + x1*np.cos(theta1), x1*np.sin(theta1), 0])
    pos2 = np.array([d0 + x2*np.cos(theta2), x2*np.sin(theta2), 0])

    pos = np.array([pos1, pos0, pos2])

    a = Atoms('3He',
              positions=pos,
              pbc=[0,0,0],
              cell=[3,3,3])
    return a

def structure_list(x1_list, x2_list):
    a_list = [structure(x1, x2) for x1, x2 in zip(x1_list, x2_list)]
    return a_list

class doubleLJ_delta():
    def __init__(self, frac):
        self.frac = frac

    def energy(self, a):
        return self.frac * E_doubleLJ(a)

def plotStruct(a, x0=0, y0=0, color='r'):
    pos = a.get_positions()[:,:2]
    pos += np.array([x0,y0])
    plt.scatter(pos[:,0], pos[:,1], c=color, marker='o', s=140, edgecolors='k')
    
def rot2D(v, angle):
    angle = angle*np.pi/180
    rotMat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    return (rotMat @ v.T).T

def make_arrow(p1,p2, width, head_width, head_length, stop_before):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vec = p2 - p1
    diff = vec - (head_length + stop_before) * vec/np.linalg.norm(vec)
    plt.arrow(p1[0], p1[1], diff[0], diff[1], width=width, head_width=head_width, head_length=head_length, fc='k', ec='k')

def plotCoordinateExample(x0,y0, scale=1, fontsize=15):
    pos = scale*cartesian_coord(0,0) + np.array([x0,y0])

    L_arrow = 1.5*scale
    arrow = L_arrow*np.array([1,0])
    arrow1 = rot2D(arrow, -60)
    arrow2 = rot2D(arrow, -120)
    plt.scatter(pos[:,0], pos[:,1], c='r', marker='o', s=140, edgecolors='k')
    p1 = pos[0,:]
    p2 = pos[2,:]
    make_arrow(p1,p1+arrow1, width=0.05, head_width=0.2, head_length=0.4, stop_before=0.00)
    make_arrow(p2,p2+arrow2, width=0.05, head_width=0.2, head_length=0.4, stop_before=0.00)
    plt.text((p1+arrow1)[0]-1.1, (p1+arrow1)[1]-0.2, 'x1', fontsize=fontsize)
    plt.text((p2+arrow2)[0]+0.3, (p2+arrow2)[1]-0.2, 'x2', fontsize=fontsize)

