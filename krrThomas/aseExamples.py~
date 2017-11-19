import numpy as np
from itertools import product

from ase.io import read, write
from ase.visualize import view

def loadStructure():
    atoms = read('fromThomas/data_SnO.traj', index=':')
    structure = atoms[0]
    return structure


if __name__ == "__main__":
    struct = loadStructure()
    pbc = struct.get_pbc()
    cell = struct.get_cell()
    Natoms = struct.get_number_of_atoms()
    atomic_numbers = struct.get_atomic_numbers() # num
    atomic_types = sorted(list(set(atomic_numbers)))
    atomic_count = [list(atomic_numbers).count(i) for i in atomic_types]
    volume = struct.get_volume()
    print(pbc)
    print(cell)
    print(Natoms)
    print(atomic_numbers)
    print(atomic_types)
    print(atomic_count)
    print(volume)

    norm1 = np.linalg.norm(cell[:,0]+cell[:,1]+cell[:,2])
    norm2 = np.linalg.norm(cell)
    print(norm1)
    print(norm2)
