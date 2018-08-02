import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature import Angular_Fingerprint
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint as Angular_Fingerprint_cy
from custom_calculators import doubleLJ_calculator

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

from time import time

dim = 3

L = 2
d = 1
pbc = [0,0,0]

"""
N = 2
x = np.array([0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

N = 3
x = np.array([1.5*L, 0.2*L, d/2,
              0.5*L, 0.9*L, d/2,
              -0.5*L, 0.5*L, d/2,])
positions = x.reshape((-1,dim))
atomtypes = ['He', 'H', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

x = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 1.5, 1, 0])
positions = x.reshape((-1,dim))
a = Atoms('H4',
          positions=positions,
          cell=[4,2,1],
          pbc=[0, 0, 0])

N = 4
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
N = 5
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2,
              0.9*L, 0.1*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'He', 'O', 'H', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
atoms = read('graphene_data/graphene_all2.traj', index=':')
a = atoms[100]
atomtypes = a.get_atomic_numbers()
N = len(a.get_atomic_numbers())
x = a.get_positions().reshape(-1)
"""

#view(a)

from delta import delta as delta_cy
from delta_py import delta as delta_py
from ase.data import covalent_radii
from ase.ga.utilities import closest_distances_generator

num = a.get_atomic_numbers()
atomic_types = sorted(list(set(num)))
print(atomic_types)
blmin = closest_distances_generator(atomic_types,                                                                                          
                                    ratio_of_covalent_radii=0.7)
print(blmin)

cov_dist = 1
deltaFunc_cy = delta_cy(cov_dist=cov_dist)
deltaFunc_py = delta_py(cov_dist=cov_dist)

print('E_py=', deltaFunc_py.energy(a))
print('E_cy=', deltaFunc_cy.energy(a))

print('F_py=', deltaFunc_py.forces(a))
print('F_cy=', deltaFunc_cy.forces(a))

