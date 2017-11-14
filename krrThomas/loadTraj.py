from ase.io import read, write
import numpy as np

atoms = read('work_folder/all.traj', index=':')
atoms = atoms[]
Na = 24
dim = 3
Ntraj = len(atoms)

pos = np.zeros((Ntraj,Na,dim))
E = np.zeros(Ntraj)
for i, a in enumerate(atoms):
    pos[i] = a.positions
    E[i] = a.get_potential_energy()

print(E.shape)
print(np.max(E)) 
