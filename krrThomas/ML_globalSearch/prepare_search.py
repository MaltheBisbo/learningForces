#!/usr/bin/env python

from ase.ga.data import PrepareDB
from startgenerator_new import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.io import read, write
from ase.visualize import view
from ase.constraints import FixAtoms
from ase import Atoms,Atom
import numpy as np
import os, sys, shutil


db_file = 'gadb.db'

if os.path.isfile(db_file):
    print('Warning about to delete old database. Are you sure you want that?")')
    yes = set(['yes','y', 'ye'])
    no = set(['no','n'])
    choice = input().lower()
    if choice in yes:
        os.remove(db_file)
        try:
            shutil.rmtree('work_folder')
        except:
            pass
    elif choice in no:
        exit()
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


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
        
# size of start population
n = 20

if True:
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

    # starting_population = [sg.get_new_candidate() for i in xrange(n)]
    starting_population = []
    for i in range(n):
        print('looking for candidate {}'.format(i))
        starting_population.append(sg.get_new_candidate())

else:
    starting_population = read('start_pop.traj@:')

# create the database to store information in
db = PrepareDB(db_file_name=db_file,
               simulation_cell=slab,
               stoichiometry=atom_numbers,)
               # population_size=population_size)

for a in starting_population:
    db.add_unrelaxed_candidate(a)
write('start_pop.traj',starting_population)

