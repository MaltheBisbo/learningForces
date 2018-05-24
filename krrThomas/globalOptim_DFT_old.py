import numpy as np

from startgenerator import StartGenerator
from startgenerator2d import StartGenerator as StartGenerator2d
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
    traj = Trajectory(label+'_lcao.traj','a', structure)
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < 1:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 5)
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
              txt = label+'_single_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)
    
    return structure.get_potential_energy()

def relax_VarianceBreak(structure, calc, label, niter_max=10):
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

    traj = Trajectory(label+'.traj','a', structure)
    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        traj.write(structure)
        return structure
    
    for niter in range(niter_max):
        if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
            return structure
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 500)
        niter += 1

    return structure


def rattle_atom2d(struct, index_rattle, rmax_rattle=1.0, rmin=0.9, rmax=1.7, Ntries=10):
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

def rattle_Natoms2d(struct, Nrattle, rmax_rattle=1.0, rmin=0.9, rmax=1.7, Ntries=10):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_rattle = np.random.permutation(Natoms)
    i_rattle = i_rattle[:Nrattle]
    
    rattle_counter = 0
    for index in i_permuted:
        newStruct = rattle_atom2d(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter > Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle

def rattle_atom2d_center(struct, index_rattle, rmax_rattle=1.0, rmin=0.9, rmax=1.7, Ntries=10):
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
    rmax = 1.3*2*cov_radii

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
    def __init__(self, calculator, traj_namebase, MLmodel=None, Natoms=6, Niter=50, rattle_maxDist=0.5, kbT=1, Nstag=10, min_saveDifference=0.3, minSampleStep=10, MLerrorMargin=0.1, Nstart_pop=5, maxNtrain=1.5e3, fracPerturb=0.3, noZ=False):

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
        self.Nperturb = max(2, int(np.ceil(self.Natoms*fracPerturb)))
        self.noZ = noZ

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0

        # Trajectory names
        self.writer_initTrain = TrajectoryWriter(filename=traj_namebase+'initTrain.traj', mode='a')
        self.writer_spTrain = TrajectoryWriter(filename=traj_namebase+'spTrain.traj', mode='a')
        self.writer_spPredict = TrajectoryWriter(filename=traj_namebase+'spPredict.traj', mode='a')
        self.writer_current = TrajectoryWriter(filename=traj_namebase+'current.traj', mode='a')

        # make txt file
        open(traj_namebase + 'sigma.txt', 'a').close()

        self.comm = world.new_communicator(np.array(range(world.size)))
        self.master = self.comm.rank == 0
        
    def runOptimizer(self):

        # Initial structures
        self.E = np.inf
        for i in range(self.Nstart_pop):
            a_init = createInitalStructure2d(self.Natoms)
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

        a_new = self.a.copy()
        pos_new = np.empty((self.Natoms, 3))
        """
        pos_new = a_new.positions
        print('0 - {}: {}'.format(world.rank, a_new.positions[0,:]))
        sys.stdout.flush()
        world.barrier()
        """
        # Run global search
        stagnation_counter = 0
        for i in range(self.Niter):
            # Perturb current structure

            a_new = rattle_Natoms2d_center(struct=a_new,
                                           Nrattle=self.Nperturb,
                                           rmax_rattle=self.rattle_maxDist)
            if world.rank == 0:
                pos_new[0] = 1
#                pos_new = a_new.positions
                print('master post rattle:', pos_new)
#                sys.stdout.flush()
            if world.rank == 1:
                print('1', pos_new)
#                sys.stdout.flush()
            world.broadcast(pos_new, 0)
#            a_new.positions = pos_new
            world.barrier()
            print('2 - {}: {}'.format(world.rank, pos_new))
            sys.stdout.flush()
           

            # Use MLmodel - if it exists
            if self.MLmodel is not None:
                if self.master:
                    print('begin training')
                    # Train ML model if new data is available
                    if len(self.a_add) > 0:
                        self.trainModel()
                    
                    print('training done')
                    # Relax with MLmodel
                    a_new, EnewML = self.relaxML(a_new)
                    print('MLrelaxing done')
                    pos_new = a_new.positions
                world.barrier()
                world.broadcast(pos_new, 0)
                a_new.positions = pos_new
                world.barrier()
                
                # Singlepoint with objective potential
                Enew = self.singlePoint(a_new)
                
            else:
                # Relax with true potential
                a_new, Enew = self.relaxTrue(a_new)

            if self.master:
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
            
    def trainModel(self):
        """
        # Reduce training data - If there is too much
        if self.ksaved > self.maxNtrain:
            Nremove = self.ksaved - self.maxNtrain
            self.ksaved = self.maxNtrain
            self.MLmodel.remove_data(Nremove)
        """
        print('train1', world.rank)
        GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1, 3, 5)}
        #GSkwargs = {'reg': [1e-5], 'sigma': [10]}
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         k=5,
                                         **GSkwargs)
        print('train2', world.rank)
        self.a_add = []
        with open(self.traj_namebase + 'sigma.txt', 'a') as f:
            f.write('{0:.2f}\n'.format(params['sigma']))
            print('train3', world.rank)

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
        # fix atons in xy-plane if noZ=True
        if self.noZ:
            plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
            a.set_constraint(plane)

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        krr_calc = krr_calculator(self.MLmodel)
        if self.master:
            a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=2)

        self.traj_counter += 1
        Erelaxed = a_relaxed.get_potential_energy()
        return a_relaxed, Erelaxed

    def relaxTrue(self, a):
        # fix atons in xy-plane if noZ=True
        if self.noZ:
            plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
            a.set_constraint(plane)

        pos = a.positions
        if self.master:
            pos = a.positions
        world.broadcast(pos, 0)
        a.positions = pos
        world.barrier()
            
        # Relax
        label = self.traj_namebase + '{}'.format(self.traj_counter)
        a_relaxed = relaxGPAW(a, label)
        if self.master:
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
        world.broadcast(pos, 0)
        a.positions = pos
        world.barrier()

        label = self.traj_namebase + '{}'.format(traj_counter)
        E = singleGPAW(a, label)
        #Calculate a and save structure with target energy
        a.energy = E
        #results = {'energy': E}
        #calc = SinglePointCalculator(a, **results)
        #a.set_calculator(calc)
        self.a_add.append(a)
        # Save to spTrain-trajectory
        if self.master:
            self.writer_spTrain.write(a, energy=E)
        self.ksaved += 1
        return E

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
    
    Natoms = 10
    
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

    # Calculator
    #calc = doubleLJ_calculator(noZ=True)
    calc = Dftb(label='C',
                Hamiltonian_SCC='No',
                #            kpts=(1,1,1),   # Hvis man paa et tidspunkt skal bruge periodiske graensebetingelser
                Hamiltonian_Eigensolver='Standard {}',
                Hamiltonian_MaxAngularMomentum_='',
                Hamiltonian_MaxAngularMomentum_C='"p"',
                Hamiltonian_Charge='0.000000',
                Hamiltonian_Filling ='Fermi {',
                Hamiltonian_Filling_empty= 'Temperature [Kelvin] = 0.000000')

    optimizer = globalOptim(calculator=calc,
                            traj_namebase=savefiles_namebase,
                            MLmodel=krr,
                            Natoms=Natoms,
                            Niter=100,
                            Nstag=100,
                            Nstart_pop=1,
                            fracPerturb=1,
                            rattle_maxDist=3,
                            kbT=0.5,
                            noZ=True)

    optimizer.runOptimizer()
