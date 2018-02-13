import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.emt import EMT

atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[10]
print('Ndata:', Ndata)
#view(a0)
E = np.array([a.get_potential_energy() for a in atoms])
print(a0.get_potential_energy())
calculator = a0.get_calculator()
a0.set_calculator(calculator)
print(a0.get_potential_energy())

l = 2
b = 1
h = 1
H2 = Atoms('H2',
           positions=[[0.25*l,0.5*b,0.5*h],[0.75*l,0.5*b,0.5*h]],
           pbc=[0,0,0],
           cell=[l,b,h])
H2.set_calculator(EMT())
print(H2.get_potential_energy())
print(H2.get_forces())
