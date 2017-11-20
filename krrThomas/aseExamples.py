import numpy as np
from itertools import product
from scipy.spatial.distance import cdist, squareform

from ase.io import read, write
from ase.visualize import view

def loadStructure():
    atoms = read('fromThomas/data_SnO.traj', index=':')
    structure = atoms[0]
    return structure


if __name__ == "__main__":
    struct = loadStructure()

    pos = struct.positions
    pbc = struct.get_pbc()
    cell = struct.get_cell()
    Natoms = struct.get_number_of_atoms()
    atomic_numbers = struct.get_atomic_numbers()  # num
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

    pos = struct.positions

    xyz = np.array([1,3,2])
    cell_displacement = xyz @ cell
    displaced_pos = cell_displacement + pos
    print(displaced_pos)

    deltaRs = np.apply_along_axis(np.linalg.norm,1,displaced_pos-pos[0])
    deltaRs2 = cdist(pos[0].reshape((1,3)), displaced_pos)
    print(deltaRs)
    print(deltaRs2)

    kk = {type:list(atomic_numbers).count(type) for type in atomic_types}

    print(kk)
