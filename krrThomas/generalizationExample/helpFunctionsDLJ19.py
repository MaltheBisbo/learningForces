import numpy as np
from ase import Atoms
from ase.visualize import view

def get_structure(c1, c2, a0, d0=1.06):
    d0 = 1.06
    dy = d0*np.sin(np.pi/3)
    dx = d0*np.cos(np.pi/3)
    
    row1 = np.c_[d0*np.arange(3).T, np.zeros(3).T, np.zeros(3).T]
    row2 = np.c_[d0*np.arange(4).T-dx, dy*np.ones(4).T, np.zeros(4).T]
    row3 = np.c_[d0*np.arange(5).T-2*dx, 2*dy*np.ones(5).T, np.zeros(5).T]
    row4 = np.c_[d0*np.arange(4).T-dx, 3*dy*np.ones(4).T, np.zeros(4).T]
    row5 = np.c_[d0*np.arange(3).T, 4*dy*np.ones(3).T, np.zeros(3).T]
    row5[0,0] += 2*dx*(c1-1)
    row5[1,0] += 2*dx*(c2-1)
    
    positions = np.r_[row1, row2, row3, row4, row5]
    structure = a0.copy()
    structure.set_positions(positions)
    #structure = Atoms('19He', positions=positions)
    return structure

def get_structure_list(c1_list, c2_list, a0):
    a_list = [get_structure(c1, c2, a0) for c1, c2 in zip(c1_list, c2_list)]
    return a_list


if __name__ == '__main__':
    a = get_structure(c1=0, c2=0)
    c1 = [-1,0.5]
    c2 = [0,0]
    a_list = get_structure_list(c1, c2)
    view(a_list)
