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
    
    # get_neighbourcells
    Rc_max = 6
    cell_vec_norms = np.linalg.norm(cell, axis=0)
    neighbours = []
    for i in range(3):
        if pbc[i]:
            ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i])))
            neighbours.append(range(-ncellmax,ncellmax+1))
        else:
            neighbours.append([0])
    neighbourcells = []
    for x,y,z in product(*neighbours):
        neighbourcells.append((x,y,z))

    print(neighbourcells)
    xyz = neighbourcells[0]

    displacement = np.dot(cell.T,np.array(xyz).T)
    displaced_pos = pos + displacement
    
    #for i in range(Natoms):
    deltaRs = np.apply_along_axis(np.linalg.norm,1,displaced_pos-pos[0])
        
    deltaRs1 = cdist(pos, displaced_pos, metric='euclidean')[0]
    #S = np.sum()
    print(deltaRs)
    print(deltaRs1)
    
