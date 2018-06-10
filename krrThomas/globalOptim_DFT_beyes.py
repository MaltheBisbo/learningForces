import numpy as np
from time import time

from startgenerator import StartGenerator
from startgenerator2d_new import StartGenerator as StartGenerator2d
from custom_calculators import krr_calculator

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.ga.utilities import closest_distances_generator
from ase.optimize import BFGS, BFGSLineSearch
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixedPlane
from ase.ga.relax_attaches import VariansBreak
from ase.data import covalent_radii

from mutations import rattle_Natoms, rattle_Natoms_center, createInitalStructure

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak

import ase.parallel as mpi
world = mpi.world

import sys


def relaxGPAW(structure, label):
    '''
    Relax a structure and saves the trajectory based in the index i

    Parameters
    ----------
    structure : ase Atoms object to be relaxed

    i : index which the trajectory is saved under

    ranks: ranks of the processors to relax the structure 

    Returns
    -------
    structure : relaxed Atoms object
    '''

    # Create calculator
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),
              occupations=FermiDirac(0.1),
              maxiter=99,
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              kpts=(1,1,1),
              txt = label+ '_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)
    
    # loop a number of times to capture if minimization stops with high force
    # due to the VariansBreak calls
    forcemax = 0.1
    niter = 0

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    
    traj = Trajectory(label+'_lcao.traj','w', structure)
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < 1:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 10)
        niter += 1

    return structure

def singleGPAW(structure, label):
    # Create calculator
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),
              occupations=FermiDirac(0.1),
              maxiter=99,
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              kpts=(1,1,1),
              txt = label+ '_single_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)

    return structure.get_potential_energy(), structure.get_forces()

def relax_VarianceBreak(structure, calc, label='', niter_max=10):
    '''
    Relax a structure and saves the trajectory based in the index i

    Parameters
    ----------
    structure : ase Atoms object to be relaxed

    Returns
    -------
    '''

    # Set calculator 
    structure.set_calculator(calc)

    # loop a number of times to capture if minimization stops with high force
    # due to the VariansBreak calls
    forcemax = 0.1
    niter = 0

    # traj = Trajectory(label+'.traj','a', structure)
    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        #traj.write(structure)
        return structure
    
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < niter_max:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        #dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 300)
        niter += 1
        
    return structure


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

class globalOptim():
    """
    --Input--
    calculator:
    ASE calculator for calculating the true energy and forces

    MLmodel:
    Model that given training data can predict energy and gradient of a structure.
    Hint: must include a fit, predict_energy and predict_force methods.
    
    Natoms:
    Number of atoms in structure.
    
    Niter:
    Number of monte-carlo steps.

    rattle_maxDist:
    Max translation of each coordinate when perturbing the current structure to
    form a new candidate.
    
    kbT:
    Variable controling how likly it is to accept a worse structure.
    Hint: Should be on the order of the energy difference between local minima.

    Nstag:
    Max number of iterations without accepting new structure before
    the search is terminated.

    min_saveDifference:
    Defines the energy which a new trajectory point has to be lover than the previous, to be saved for training.
    
    minSampleStep:
    if minSampleStep=10, every tenth point in the relaxation trajectory is used for training, and so on..
    Unless min_saveDifference have not been ecxeeded.

    MLerrorMargin:
    Maximum error differende between the ML-relaxed structure and the target energy of the same structure,
    below which the ML structure is accepted.

    NstartML:
    The Number of training data required for the ML-model to be used.

    maxNtrain:
    The maximum number of training data. When above this, some of the oldest training data is removed.

    radiusRange:
    Range [rmin, rmax] constraining the initial and perturbed structures. All atoms need to be atleast a
    distance rmin from each other, and have atleast one neighbour less than rmax away.

    fracPerturb:
    The fraction of the atoms which are ratteled to create a new structure.

    noZ:
    Atoms are not mooved in the z-direction during relaxation.
    """
    def __init__(self, calculator, traj_namebase, MLmodel=None, Natoms=6, Niter=50, rattle_maxDist=0.5, kbT=1, Nstag=10, min_saveDifference=0.3, minSampleStep=10, MLerrorMargin=0.1, Nstart_pop=5, maxNtrain=1.5e3, Nperturb=2, noZ=False, dualPoint=False):

        self.calculator = calculator
        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel

        self.Natoms = Natoms
        self.rattle_maxDist = rattle_maxDist
        self.Niter = Niter
        self.kbT = kbT
        self.Nstag = Nstag
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.MLerrorMargin = MLerrorMargin
        #self.NstartML = NstartML
        self.Nstart_pop = Nstart_pop
        self.maxNtrain = int(maxNtrain)
        self.Nperturb = Nperturb
        self.noZ = noZ
        self.dualPoint = dualPoint

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0

        # Define parallel communication
        self.comm = world.new_communicator(np.array(range(world.size)))
        self.master = self.comm.rank == 0

        # Trajectory names
        self.writer_initTrain = Trajectory(filename=traj_namebase+'initTrain.traj', mode='a', master=self.master)
        self.writer_spTrain = Trajectory(filename=traj_namebase+'spTrain.traj', mode='a', master=self.master)
        self.writer_spPredict = Trajectory(filename=traj_namebase+'spPredict.traj', mode='a', master=self.master)
        self.writer_current = Trajectory(filename=traj_namebase+'current.traj', mode='a', master=self.master)

        # make txt file
        open(traj_namebase + 'sigma.txt', 'a').close()
        open(traj_namebase + 'MLerror_Ntries.txt', 'a').close()
        open(traj_namebase + 'E_MLerror.txt', 'a').close()
        open(traj_namebase + 'time.txt', 'a').close()

    def runOptimizer(self):

        # Initial structures
        self.E = np.inf
        for i in range(self.Nstart_pop):
            if self.noZ:
                a_init = createInitalStructure2d(self.Natoms)
            else:
                a_init = createInitalStructure(self.Natoms)
            a, E = self.relaxTrue(a_init)
            if E < self.E:
                self.a = a.copy()
                self.E = E
                
        # Reset traj_counter for ML-relaxations
        self.traj_counter = 0

        # Initialize the best structure
        if self.master:
            self.a_best = self.a.copy()
            self.Ebest = self.E

        # initialise new search structure
        a_new = self.a.copy()
        pos_new = np.zeros((self.Natoms, 3))

        
        # Run global search
        stagnation_counter = 0
        for i in range(self.Niter):
            # Use MLmodel - if it exists
            if self.MLmodel is not None:
                # Train ML model if new data is available
                if len(self.a_add) > 0:
                    self.trainModel()
                
                # Generate new rattled + MLrelaxed candidate
                t_newCand_start = time()
                a_new = self.newCandidate_beyes(self.a)
                t_newCand_end = time()
                
                # Singlepoint with objective potential
                t_sp_start = time()
                Enew, Fnew = self.singlePoint(a_new)
                t_sp_end = time()
                if self.master:
                    with open(self.traj_namebase + 'time.txt', 'a') as f:
                        f.write('{}\t{}\n'.format(t_newCand_end-t_newCand_start, t_sp_end-t_sp_start))

                if self.dualPoint and i > 50:
                    a_dp, E_dp = self.get_dualPoint(a_new, Fnew)
                    if E_dp < Enew:
                        anew = a_dp.copy
                        Enew = E_dp
                
            else:
                # Perturb current structure
                if self.master:
                    a_new = rattle_Natoms2d_center(struct=a_new,
                                                   Nrattle=self.Nperturb,
                                                   rmax_rattle=self.rattle_maxDist)
                    pos_new = a_new.positions
                self.comm.broadcast(pos_new, 0)
                a_new.positions = pos_new
                self.comm.barrier()
                
                # Relax with true potential
                a_new, Enew = self.relaxTrue(a_new)

            if self.master:
                print('iteration {0:d}: Ebest={1:.2f}, Ecur={2:.2f}, Enew={3:.2f}'.format(i, self.Ebest, self.E, Enew))
                dE = Enew - self.E
                if dE <= 0:  # Accept better structure
                    self.E = Enew
                    self.a = a_new.copy()
                    stagnation_counter = 0
                    if Enew < self.Ebest:  # Update the best structure
                        self.Ebest = Enew
                        self.a_best = a_new.copy()
                else:
                    p = np.random.rand()
                    if p < np.exp(-dE/self.kbT):  # Accept worse structure
                        self.E = Enew
                        self.a = a_new.copy()
                        stagnation_counter = 0
                    else:  # Reject structure
                        stagnation_counter += 1
                        
                # Save current structure
                self.writer_current.write(self.a, energy=self.E)
                    
                if stagnation_counter >= self.Nstag:  # The search has converged or stagnated.
                    #print('Stagnation criteria was reached')
                    break

    def get_dualPoint(self, a, F):
        max_atom_displacement = 0.08  # The atom with the largest force will be displaced this ammount

        a_dp = a.copy()
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_dp = a.positions + max_atom_displacement * F/Fmax
        
        a_dp.set_positions(pos_dp)
        
        E_dp, F_dp = self.singlePoint(a_dp)
        
        return a_dp, E_dp
        
    def mutate(self, a, Ntasks_each):
        N_newCandidates = self.comm.size * Ntasks_each
        
        N_exploit_rattle = int(0.3 * N_newCandidates)  
        N_exploit_disk = int(0.3 * N_newCandidates)  
        N_explore = N_newCandidates - N_exploit_rattle - N_exploit_disk

        mutation_tasks = [0]*3
        i_rattle = []  # save which number the core is to make the rattle mutation.
        for i in range(N_newCandidates):
            rank = i % self.comm.size
            if self.comm.rank == rank:
                if i < N_explore:
                    mutation_tasks[0] += 1
                elif i < N_explore + N_exploit_disk:
                    mutation_tasks[1] += 1
                else:
                    i_rattle.append(i - (N_explore + N_exploit_disk))
                    mutation_tasks[2] += 1
        self.comm.barrier()
        
        a_mutated_list = []
        rmax_rattle_list = np.linspace(0.2, 0.8, N_exploit_rattle)
        for i_mut, n_mut in enumerate(mutation_tasks):
            for k in range(n_mut):
                a_mutated = a.copy()
                # make 2d or 3d mutations
                if self.noZ:
                    if i_mut == 0:
                        a_mutated = createInitalStructure2d(self.Natoms)
                    elif i_mut == 1:
                        a_mutated = rattle_Natoms2d_center(struct=a_mutated,
                                                           Nrattle=self.Nperturb,
                                                           rmax_rattle=self.rattle_maxDist)
                    else:
                        rmax_rattle = rmax_rattle_list[i_rattle[k]]
                        a_mutated = rattle_Natoms2d(struct=a_mutated,
                                                    Nrattle=self.Natoms,
                                                    rmax_rattle=rmax_rattle)
                else:
                    if i_mut == 0:
                        a_mutated = createInitalStructure(self.Natoms)
                    elif i_mut == 1:
                        a_mutated = rattle_Natoms_center(struct=a_mutated,
                                                           Nrattle=self.Nperturb,
                                                           rmax_rattle=self.rattle_maxDist)
                    else:
                        rmax_rattle = rmax_rattle_list[i_rattle[k]]
                        a_mutated = rattle_Natoms(struct=a_mutated,
                                                    Nrattle=self.Natoms,
                                                    rmax_rattle=rmax_rattle)
                a_mutated_list.append(a_mutated)
        self.comm.barrier()
                
        return a_mutated_list

    
    def newCandidate_beyes(self, a):
        N_newCandidates = 50
        # the maximum number of candidates a core need to make to make N_newCandidates on a single node.
        N_tasks = int(np.ceil(N_newCandidates / self.comm.size))
        # Why not use all cores.
        N_newCandidates = N_tasks * N_newCandidates
        
        anew = a.copy()
        pos_new = a.positions  # for later
        pos_new_mutated = a.positions  # for later
        if self.master:
            print('Begin mutation')
        anew_mutated_list = self.mutate(anew,
                                        N_tasks)
        if self.master:
            print('mutations ended')
        
        # Relax with MLmodel
        anew_list = []
        E_list = []
        error_list = []
        for anew_mutated in anew_mutated_list:
            anew = self.relaxML(anew_mutated)
            anew_list.append(anew)
            
            E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
            E_list.append(E)
            error_list.append(error)
        E_list = np.array(E_list)
        error_list = np.array(error_list)
        
        # Gather data from slaves to master
        pos_new_list = np.array([anew.positions for anew in anew_list])
        pos_new_mutated_list = np.array([anew_mutated.positions for anew_mutated in anew_mutated_list])
        if self.comm.rank == 0:
            E_all = np.empty(N_tasks * self.comm.size, dtype=float)
            error_all = np.empty(N_tasks * self.comm.size, dtype=float)
            pos_all = np.empty(N_tasks * 3*self.Natoms*self.comm.size, dtype=float)
            pos_all_mutated = np.empty(N_tasks * 3*self.Natoms*self.comm.size, dtype=float)
        else:
            E_all = None
            error_all = None
            pos_all = None
            pos_all_mutated = None
        self.comm.gather(E_list, 0, E_all)
        self.comm.gather(error_list, 0, error_all)
        self.comm.gather(pos_new_list.reshape(-1), 0, pos_all)
        self.comm.gather(pos_new_mutated_list.reshape(-1), 0, pos_all_mutated)

        # Pick best candidate on master
        if self.master:
            
            EwithError_all = E_all - 2 * error_all
            index_best = EwithError_all.argmin()
            
            print('{}:\n'.format(self.traj_counter), np.c_[E_all, error_all])
            print('{} best:\n'.format(self.traj_counter), E_all[index_best], error_all[index_best])
            
            with open(self.traj_namebase + 'E_MLerror.txt', 'a') as f:
                f.write('{0:.4f}\t{1:.4f}\n'.format(E_all[index_best], error_all[index_best]))
            
            pos_all = pos_all.reshape((N_tasks * self.comm.size, self.Natoms, 3))
            pos_new = pos_all[index_best]
            pos_all_mutated = pos_all_mutated.reshape((N_tasks * self.comm.size, self.Natoms, 3))
            pos_new_mutated = pos_all_mutated[index_best]
        
        self.comm.broadcast(pos_new, 0)
        anew.positions = pos_new
        self.comm.broadcast(pos_new_mutated, 0)
        anew_mutated = a.copy()
        anew_mutated.positions = pos_new_mutated
        self.comm.barrier()

        # Write unrelaxed + relaxed versions of new candidate to file
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        write(label+'.traj', [anew_mutated, anew])

        self.traj_counter += 1
        return anew
    
    """
    def newCandidate_beyes(self, a):
        anew = a.copy()
        anew_unrelaxed = rattle_Natoms2d_center(struct=anew,
                                                Nrattle=self.Nperturb,
                                                rmax_rattle=self.rattle_maxDist)
        
        # Relax with MLmodel
        anew = self.relaxML(anew_unrelaxed)
        E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
        E = np.array(E)
        error = np.array(error)
        
        # Gather data from slaves to master
        pos_new = anew.positions  # .reshape(-1)
        pos_new_unrelaxed = anew_unrelaxed.positions  # .reshape(-1)
        if self.comm.rank == 0:
            E_all = np.empty(world.size, dtype=float)
            error_all = np.empty(world.size, dtype=float)
            pos_all = np.empty(3*self.Natoms*world.size, dtype=float)
            pos_all_unrelaxed = np.empty(3*self.Natoms*world.size, dtype=float)
        else:
            E_all = None
            error_all = None
            pos_all = None
            pos_all_unrelaxed = None
        self.comm.gather(E, 0, E_all)
        self.comm.gather(error, 0, error_all)
        self.comm.gather(pos_new.reshape(-1), 0, pos_all)
        self.comm.gather(pos_new_unrelaxed.reshape(-1), 0, pos_all_unrelaxed)

        # Pick best candidate on master
        if self.master:
            EwithError_all = E_all - 2 * error_all
            index_best = EwithError_all.argmin()
            
            print('{}:\n'.format(self.traj_counter), np.c_[E_all, error_all])
            print('{} best:\n'.format(self.traj_counter), E_all[index_best], error_all[index_best])
            
            with open(self.traj_namebase + 'E_MLerror.txt', 'a') as f:
                f.write('{0:.4f}\t{1:.4f}\n'.format(E_all[index_best], error_all[index_best]))
            
            pos_all = pos_all.reshape((self.comm.size, self.Natoms, 3))
            pos_new = pos_all[index_best]
            pos_all_unrelaxed = pos_all_unrelaxed.reshape((self.comm.size, self.Natoms, 3))
            pos_new_unrelaxed = pos_all_unrelaxed[index_best]
        self.comm.broadcast(pos_new, 0)
        anew.positions = pos_new
        self.comm.broadcast(pos_new_unrelaxed, 0)
        anew_unrelaxed.positions = pos_new_unrelaxed
        self.comm.barrier()

        # Write unrelaxed + relaxed versions of new candidate to file
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        write(label+'.traj', [anew_unrelaxed, anew])

        self.traj_counter += 1
        return anew
    """
    
    def trainModel(self):
        """
        # Reduce training data - If there is too much
        if self.ksaved > self.maxNtrain:
            Nremove = self.ksaved - self.maxNtrain
            self.ksaved = self.maxNtrain
            self.MLmodel.remove_data(Nremove)
        """
        GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1, 3, 5)}
        #GSkwargs = {'reg': [1e-5], 'sigma': [10]}
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         k=5,
                                         **GSkwargs)
        self.a_add = []
        if self.master:
            with open(self.traj_namebase + 'sigma.txt', 'a') as f:
                f.write('{0:.2f}\n'.format(params['sigma']))

    def add_trajectory_to_training(self, trajectory_file):
        atoms = read(filename=trajectory_file, index=':')
        E = [a.get_potential_energy() for a in atoms]
        Nstep = len(atoms)

        # Always add+save start structure
        self.a_add.append(atoms[0])
        self.writer_initTrain.write(atoms[0], energy=E[0])

        n_last = 0
        Ecurrent = E[0]
        for i in range(1,Nstep-int(self.minSampleStep/2)):
            n_last += 1
            if Ecurrent - E[i] > self.min_saveDifference and n_last > self.minSampleStep:
                self.a_add.append(atoms[i])
                Ecurrent = E[i]
                self.ksaved += 1
                n_last = 0
                
                # Save to initTrain-trajectory
                self.writer_initTrain.write(atoms[i], energy=E[i])

        # Always save+add last structure
        self.a_add.append(atoms[-1])
        self.writer_initTrain.write(atoms[-1], energy=E[-1])
                
    def relaxML(self, a):
        # fix atoms in xy-plane if noZ=True
        if self.noZ:
            plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
            a.set_constraint(plane)

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        krr_calc = krr_calculator(self.MLmodel)
        a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=2)

        #self.traj_counter += 1
        return a_relaxed

    def relaxTrue(self, a):
        # fix atons in xy-plane if noZ=True
        if self.noZ:
            plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
            a.set_constraint(plane)

        pos = a.positions
        if self.master:
            pos = a.positions
        self.comm.broadcast(pos, 0)
        a.positions = pos
        self.comm.barrier()

        # Relax
        label = self.traj_namebase + '{}'.format(self.traj_counter)
        a_relaxed = relaxGPAW(a, label)

        # Add sampled trajectory to training data.
        self.add_trajectory_to_training(label+'_lcao.traj')

        self.traj_counter += 1
        Erelaxed = a_relaxed.get_potential_energy()
        return a_relaxed, Erelaxed
            
    def singlePoint(self, a):
        # Check if datapoint is new based on KRR prediction error
        #E, error, _ = self.MLmodel.predict_energy(atoms=a, return_error=True)
        #if error < 0.05:
        #    return E
        
        # Save structure with ML-energy
        if self.master:
            self.writer_spPredict.write(a)

        pos = a.positions
        if self.master:
            pos = a.positions
        self.comm.broadcast(pos, 0)
        a.positions = pos
        self.comm.barrier()
        
        label =  self.traj_namebase + '{}'.format(self.traj_counter)
        E, F = singleGPAW(a, label)
        self.comm.barrier()
        
        a.energy = E
        self.a_add.append(a)

        # Save to spTrain-trajectory
        self.writer_spTrain.write(a, energy=E, forces=F)
        self.ksaved += 1
        return E, F

if __name__ == '__main__':
    from custom_calculators import doubleLJ_calculator
    from gaussComparator import gaussComparator
    #from angular_fingerprintFeature import Angular_Fingerprint
    from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
    from krr_ase import krr_class
    #from krr_fracTrain import krr_class
    from delta_functions.delta import delta as deltaFunc
    from ase.calculators.dftb import Dftb
    import sys

    from ase.data import covalent_radii
    
    Natoms = 24
    
    # Set up featureCalculator
    a = createInitalStructure(Natoms)

    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 5
    use_angular = True
    
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    
    
    # Set up KRR-model
    comparator = gaussComparator()
    delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator,
                    delta_function=delta_function)

    #krr = krr_class(comparator=comparator,
    #                featureCalculator=featureCalculator,
    #                delta_function=delta_function,
    #                fracTrain=0.8)
    

    # Savefile setup
    savefiles_path = sys.argv[1]
    try:
        run_num = sys.argv[2]
    except IndexError:
        run_num = ''
    savefiles_namebase = savefiles_path + 'global' + run_num + '_' 

    optimizer = globalOptim(calculator=None,
                            traj_namebase=savefiles_namebase,
                            MLmodel=krr,
                            Natoms=Natoms,
                            Niter=600,
                            Nstag=600,
                            Nstart_pop=2,
                            Nperturb=2,
                            rattle_maxDist=5,
                            kbT=0.5,
                            noZ=False,
                            dualPoint=True)

    optimizer.runOptimizer()
