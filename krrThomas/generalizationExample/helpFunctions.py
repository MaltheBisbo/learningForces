import numpy as np
from doubleLJ import doubleLJ_energy_ase as E_doubleLJ

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
