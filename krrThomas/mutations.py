import numpy as np
from time import time

from startgenerator_new import StartGenerator
from startgenerator2d_new import StartGenerator as StartGenerator2d

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.ga.utilities import closest_distances_generator
from ase.constraints import FixedPlane
from ase.ga.relax_attaches import VariansBreak
from ase.data import covalent_radii


def rattle_atom(struct, index_rattle, rmax_rattle=1.0, rmin=1.0, rmax=1.7, Ntries=50):
    Natoms = struct.get_number_of_atoms()
    
    structRattle = struct.copy()
    mindis = 0
    mindisAtom = 10

    for i in range(Ntries):
        # First load original positions
        positions = struct.positions.copy()
        
        # Then Rattle within a circle
        r = rmax_rattle * np.random.rand()**(1/3)
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        positions[index_rattle] += r * np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])

        structRattle.positions = positions
        dis = structRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])  # Check that we are not too close
        mindisAtom = np.min(structRattle.get_distances(index_rattle, np.delete(np.arange(Natoms), index_rattle)))  # check that we are not too far
        """
        # If it does not fit, try to wiggle it into place using small circle
        if mindis < rmin:
            for i in range(10):
                r = 0.3 * np.random.rand()**(1/3)
                theta = np.random.uniform(low=0, high=2 * np.pi)
                positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])
                structRattle.positions = positions
                dis = structRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > rmin:
                    break

                # Otherwise reset coordinate
                else:
                    positions[index_rattle] -= r * np.array([np.cos(theta), np.sin(theta), 0])
        """
        # STOP CRITERION
        if mindis > rmin and mindisAtom < rmax:
            return structRattle
    
    # Return None if no acceptable rattle was found
    return None

def rattle_Natoms(struct, Nrattle, rmax_rattle=1.0, Ntries=50):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_rattle = np.random.permutation(Natoms)
    i_rattle = i_rattle[:Nrattle]

    cov_radii = covalent_radii[6] # cd[(6,6)]  # hard coded
    rmin = 0.7*2*cov_radii
    rmax = 1.1*2*cov_radii
    
    rattle_counter = 0
    for index in i_rattle:
        newStruct = rattle_atom(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter > Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle


def rattle_atom2d(struct, index_rattle, rmax_rattle=1.0, rmin=1.0, rmax=1.7, Ntries=10):
    Natoms = struct.get_number_of_atoms()
    
    structRattle = struct.copy()
    mindis = 0
    mindisAtom = 10

    for i in range(Ntries):
        # First load original positions
        positions = struct.positions.copy()
        
        # Then Rattle within a circle
        r = rmax_rattle * np.sqrt(np.random.rand())
        theta = np.random.uniform(low=0, high=2*np.pi)
        positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])

        structRattle.positions = positions
        dis = structRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])  # Check that we are not too close
        mindisAtom = np.min(structRattle.get_distances(index_rattle, np.delete(np.arange(Natoms), index_rattle)))  # check that we are not too far
        
        # If it does not fit, try to wiggle it into place using small circle
        if mindis < rmin:
            for i in range(10):
                r = 0.5 * np.sqrt(np.random.rand())
                theta = np.random.uniform(low=0, high=2 * np.pi)
                positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])
                structRattle.positions = positions
                dis = structRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > rmin:
                    break

                # Otherwise reset coordinate
                else:
                    positions[index_rattle] -= r * np.array([np.cos(theta), np.sin(theta), 0])

        # STOP CRITERION
        if mindis > rmin and mindisAtom < rmax:
            return structRattle
    
    # Return None if no acceptable rattle was found
    return None


def rattle_Natoms2d(struct, Nrattle, rmax_rattle=1.0, Ntries=10):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_rattle = np.random.permutation(Natoms)
    i_rattle = i_rattle[:Nrattle]

    cov_radii = covalent_radii[6] # cd[(6,6)]  # hard coded
    rmin = 0.7*2*cov_radii
    rmax = 1.1*2*cov_radii
    
    rattle_counter = 0
    for index in i_rattle:
        newStruct = rattle_atom2d(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter > Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle

def rattle_atom_center(struct, index_rattle, rmax_rattle=1.0, rmin=1.0, rmax=1.6, Ntries=10):
    Natoms = struct.get_number_of_atoms()
    
    structRattle = struct.copy()
    mindis = 0
    mindisAtom = 10

    # Get unit-cell center
    center = struct.cell.sum(axis=0)/2

    for i in range(Ntries):
        # First load original positions
        positions = struct.positions.copy()
        
        # Randomly chooce rattle range and angle
        r = rmax_rattle * np.sqrt(np.random.rand())
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        
        # Apply rattle from center
        positions[index_rattle] = center
        positions[index_rattle] += r * np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])

        structRattle.positions = positions
        dis = structRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])  # Check that we are not too close
        mindisAtom = np.min(structRattle.get_distances(index_rattle, np.delete(np.arange(Natoms), index_rattle)))  # check that we are not too far
        
        # If it does not fit, try to wiggle it into place using small circle
        if mindis < rmin:
            for i in range(10):
                r = 0.5 * np.sqrt(np.random.rand())
                theta = np.random.uniform(low=0, high=2 * np.pi)
                positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])
                structRattle.positions = positions
                dis = structRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > rmin:
                    break

                # Otherwise reset coordinate
                else:
                    positions[index_rattle] -= r * np.array([np.cos(theta), np.sin(theta), 0])

        # STOP CRITERION
        if mindis > rmin and mindisAtom < rmax:
            return structRattle
    
    # Return None if no acceptable rattle was found
    return None

def rattle_Natoms_center(struct, Nrattle, rmax_rattle=5.0, Ntries=50):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_permuted = np.random.permutation(Natoms)
    atom_numbers = struct.get_atomic_numbers()

    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)
    
    cov_radii = covalent_radii[6] # cd[(6,6)]  # hard coded
    rmin = 0.7*2*cov_radii
    rmax = 1.1*2*cov_radii

    rattle_counter = 0
    for index in i_permuted:
        newStruct = rattle_atom_center(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter >= Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle

def rattle_atom2d_center(struct, index_rattle, rmax_rattle=1.0, rmin=1.0, rmax=1.6, Ntries=10):
    Natoms = struct.get_number_of_atoms()
    
    structRattle = struct.copy()
    mindis = 0
    mindisAtom = 10

    # Get unit-cell center
    center = struct.cell.sum(axis=0)/2

    for i in range(Ntries):
        # First load original positions
        positions = struct.positions.copy()
        
        # Randomly chooce rattle range and angle
        r = rmax_rattle * np.sqrt(np.random.rand())
        theta = np.random.uniform(low=0, high=2*np.pi)

        # Apply rattle from center
        positions[index_rattle] = center
        positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])

        structRattle.positions = positions
        dis = structRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])  # Check that we are not too close
        mindisAtom = np.min(structRattle.get_distances(index_rattle, np.delete(np.arange(Natoms), index_rattle)))  # check that we are not too far
        
        # If it does not fit, try to wiggle it into place using small circle
        if mindis < rmin:
            for i in range(10):
                r = 0.5 * np.sqrt(np.random.rand())
                theta = np.random.uniform(low=0, high=2 * np.pi)
                positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])
                structRattle.positions = positions
                dis = structRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > rmin:
                    break

                # Otherwise reset coordinate
                else:
                    positions[index_rattle] -= r * np.array([np.cos(theta), np.sin(theta), 0])

        # STOP CRITERION
        if mindis > rmin and mindisAtom < rmax:
            return structRattle
    
    # Return None if no acceptable rattle was found
    return None

def rattle_Natoms2d_center(struct, Nrattle, rmax_rattle=5.0, Ntries=20):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_permuted = np.random.permutation(Natoms)
    atom_numbers = struct.get_atomic_numbers()

    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)
    
    cov_radii = covalent_radii[6] # cd[(6,6)]  # hard coded
    rmin = 0.7*2*cov_radii
    rmax = 1.1*2*cov_radii

    rattle_counter = 0
    for index in i_permuted:
        newStruct = rattle_atom2d_center(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter > Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle

def createInitalStructure(Natoms):
    '''
    Creates an initial structure of 24 Carbon atoms
    '''    
    number_type1 = 6  # Carbon
    number_opt1 = Natoms  # number of atoms
    atom_numbers = number_opt1 * [number_type1]

    cell = np.array([[24, 0, 0],
                     [0, 24, 0],
                     [0, 0, 24]])
    pbc = [False, False, False]

    template = Atoms('')
    template.set_cell(cell)
    template.set_pbc(pbc)
    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a a center position (p0)
    # and three spanning vectors
    
    a = np.array((4.0, 0., 0.))
    b = np.array((0, 4.0, 0))
    z = np.array((0, 0, 4.0))
    p0 = np.array((10., 10., 10.))
    
    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)

    # create the start structure
    sg = StartGenerator(slab=template,
                        atom_numbers=atom_numbers,
                        closest_allowed_distances=cd,
                        box_to_place_in=[p0, [a, b, z]],
                        elliptic=False,
                        cluster=True)

    structure = sg.get_new_candidate(maxlength=1.6)
    return structure

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

    a = np.array((7, 0., 0.))  # 4.5 for N=10, 6 for N=24
    b = np.array((0, 7, 0))
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
                          cluster=True)

    structure = sg.get_new_candidate(maxlength=1.6)
    return structure


if __name__ == '__main__':
    from ase.visualize import view

    Natoms = 4
    a = createInitalStructure(Natoms)
    a_rattled = rattle_Natoms(a, Natoms)
    a_ratCenter = rattle_Natoms_center(a_rattled, 1)
    view([a, a_rattled, a_ratCenter])
    

    """
    atoms_list = []
    for i in range(10):
        print(i)
        a = createInitalStructure(Natoms)
        atoms_list.append(a)
    view(atoms_list)
    """
