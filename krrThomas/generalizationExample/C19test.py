import numpy as np
from ase import Atoms
from ase.visualize import view

d0 = 1.06
dy = d0*np.sin(np.pi/3)
dx = d0*np.cos(np.pi/3)

row1 = np.c_[d0*np.arange(3).T, np.zeros(3).T, np.zeros(3).T]
row2 = np.c_[d0*np.arange(4).T-dx, dy*np.ones(4).T, np.zeros(4).T]
row3 = np.c_[d0*np.arange(5).T-2*dx, 2*dy*np.ones(5).T, np.zeros(5).T]
row4 = np.c_[d0*np.arange(4).T-dx, 3*dy*np.ones(4).T, np.zeros(4).T]
row5 = np.c_[d0*np.arange(3).T, 4*dy*np.ones(3).T, np.zeros(3).T]

positions = np.r_[row1, row2, row3, row4, row5]
structure = Atoms('19He', positions=positions)
view(structure)
