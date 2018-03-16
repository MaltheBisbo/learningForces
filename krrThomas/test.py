import ase
from ase.ga.startgenerator import StartGenerator

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

atoms = read('graphene_data/graphene_all2.traj', index=':')
a = atoms[0]
atom_numbers = [1,1,1,1,1,1,1,1]
closest_allowed_distances = 1.0

startgenerator = StartGenerator(slab=a, atom_numbers=atom_numbers,
                                closest_allowed_distances=closest_allowed_distances)

a0 = startgenerator.get_new_candidate()
view(a0)


