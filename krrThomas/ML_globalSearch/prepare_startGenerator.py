#!/usr/bin/env python

from startgenerator_new import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase import Atoms,Atom
import numpy as np

def prepare_startGenerator():
    
    cell = np.array([[20, 0, 0],
                     [0, 20, 0],
                     [0, 0, 20]])
    pbc = [False, False, False]
    
    # Define template
    slab = Atoms('',
                 cell=cell,
                 pbc=pbc)
    
    # Define the box to place the atoms within
    # The volume is defined by a corner position (p0) and three spanning vectors (v)
    v = np.array([[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]])
    p0 = np.diag((cell-v)/2)
    
    # Define the composition of the atoms to optimize
    atom_numbers = [6]*24  # 24 carbon atoms
    
    # define the closest distance two atoms of a given species can be to each other
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    cd = closest_distances_generator(atom_numbers=unique_atom_types,
                                     ratio_of_covalent_radii=0.7)
    # create the starting population
    sg = StartGenerator(slab = slab,
                        atom_numbers = atom_numbers, 
                        closest_allowed_distances = cd,
                        box_to_place_in = [p0, v],
                        elliptic = False,
                        cluster = True)
        
    return sg
