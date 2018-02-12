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
view(a0)
E = np.array([a.get_potential_energy() for a in atoms])
print(a0.get_potential_energy())
calculator = a0.get_calculator()
print(calculator)
sp_calc = SinglePointCalculator
calc = EMT()
print(calc)
#a0.set_calculator(calc)
#a0.get_potential_energy()
