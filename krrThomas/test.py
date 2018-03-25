import numpy as np
from startgenerator import StartGenerator
from startgenerator2d import StartGenerator as StartGenerator2d
from custom_calculators import doubleLJ_calculator

from ase import Atoms
from ase.visualize import view
from ase.io import read, write
from ase.ga.utilities import closest_distances_generator
from ase.optimize import BFGS
from ase.constraints import FixedPlane
from ase.calculators.dftb import Dftb
from ase.constraints import FixedPlane


def createInitalStructure2d(Natoms):
    '''
    Creates an initial structure of 24 Carbon atoms
    '''    
    number_type1 = 6  # Carbon
    number_opt1 = Natoms  # number of atoms
    atom_numbers = number_opt1 * [number_type1]

    cell = np.array([[24, 0, 0],
                     [0, 24, 0],
                     [0, 0, 18]])
    pbc = [False, False, False]

    template = Atoms('')
    template.set_cell(cell)
    template.set_pbc(pbc)
    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a a center position (p0)
    # and three spanning vectors
    
    a = np.array((4.5, 0., 0.))
    b = np.array((0, 4.5, 0))
    center = np.array((11.5, 11.5, 9))
    
    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)

    # create the start structure
    sg = StartGenerator2d(slab=template,
                          atom_numbers=atom_numbers,
                          closest_allowed_distances=cd,
                          plane_to_place_in=[[a, b], center],
                          elliptic=False,
                          cluster=False)

    structure = sg.get_new_candidate()
    return structure

import sys

def createInitalStructure():
    '''
    Creates an initial structure of 24 Carbon atoms
    '''
    
    number_type1 = 6  # Carbon
    number_opt1 = 24  # number of atoms
    atom_numbers = number_opt1 * [number_type1]

    cell = np.array([[24, 0, 0],
                     [0, 24, 0],
                     [0, 0, 18]])
    pbc = [False, False, False]

    template = Atoms('')
    template.set_cell(cell)
    template.set_pbc(pbc)
    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a a center position (p0)
    # and three spanning vectors
    
    a = np.array((4.5, 0., 0.))
    b = np.array((0, 4.5, 0))
    z = np.array((0, 0, 1.5))
    p0 = np.array((0., 0., 9-0.75))
    center = np.array((11.5, 11.5))
    
    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)

    # create the start structure
    sg = StartGenerator(slab=template,
                        atom_numbers=atom_numbers,
                        closest_allowed_distances=cd,
                        box_to_place_in=[p0, [a, b, z], center],
                        elliptic=True,
                        cluster=False)

    structure = sg.get_new_candidate()
    return structure

def makeStructure(Natoms):
    dim = 3
    boxsize = 2 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 2.2

    def validPosition(X, xnew):
        Natoms = int(len(X)/dim)  # Current number of atoms
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[dim*i:dim*(i+1)])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    X = np.zeros(dim*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.r_[np.random.rand(dim-1) * boxsize, boxsize/2]
            if validPosition(X[:dim*i], xnew):
                X[dim*i:dim*(i+1)] = xnew
                break
    X = X.reshape((-1, 3))

    atomtypes = str(Natoms) + 'He'
    pbc = [0,0,0]
    cell = [boxsize]*3
    a = Atoms(atomtypes,
              positions=X,
              pbc=pbc,
              cell=cell)
    return a


<<<<<<< HEAD
atoms = createInitalStructure2d(13)
#atoms1 = makeStructure(20)
=======
atoms = createInitalStructure()

>>>>>>> 13e29e4863d4fa3f9e97202047ae5d4dbc6756b7

calc = Dftb(label='C',
            Hamiltonian_SCC='No',
#            kpts=(1,1,1),   # Hvis man paa et tidspunkt skal bruge periodiske graensebetingelser
            Hamiltonian_Eigensolver='Standard {}',
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_C='"p"',
            Hamiltonian_Charge='0.000000',
            Hamiltonian_Filling ='Fermi {',
            Hamiltonian_Filling_empty= 'Temperature [Kelvin] = 0.000000')

atoms.set_calculator(calc)
poss = atoms.positions
for pos in poss:
    pos[2] = 9

atoms.set_positions(poss)
plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(atoms))]
atoms.set_constraint(plane)
dyn = BFGS(atoms, trajectory='testLJ.traj')
dyn.run(fmax=0.1)




