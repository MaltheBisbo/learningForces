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
from populationMC import population

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak

import ase.parallel as mpi
world = mpi.world

import sys

def relaxGPAW(structure, label, forcemax=0.1, niter_max=1, steps=10):
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
    niter = 0

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    
    traj = Trajectory(label+'_lcao.traj','w', structure)
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < niter_max:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = steps)
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

def relax_VarianceBreak(structure, calc, label='', niter_max=10, forcemax=0.1):
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

class globalOptim():
    """
    --Input--
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
    
    min_saveDifference:
    Defines the energy which a new trajectory point has to be lover than the previous, to be saved for training.
    
    minSampleStep:
    if minSampleStep=10, every tenth point in the relaxation trajectory is used for training, and so on..
    Unless min_saveDifference have not been ecxeeded.

    """
    def __init__(self, traj_namebase, MLmodel, a0, mutationSelector, population_size=5, kappa=2, Niter=50, Ninit=2, min_saveDifference=0.3, minSampleStep=10, dualPoint=False):

        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel
        self.mutationSelector = mutationSelector

        
        
        self.kappa = kappa
        self.Natoms = a0.get_number_of_atoms()
        self.Niter = Niter
        self.Ninit = Ninit
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.dualPoint = dualPoint

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0
        
        self.population = population(population_size=population_size, comparator=self.MLmodel.comparator)
        self.population.add_structure(a0, 2, np.ones((self.Natoms,3)))
        
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
        # Reset traj_counter for ML-relaxations
        self.traj_counter = 0

        # Run global search
        for i in range(self.Niter):
            print('search iteration:', i)

            # Train ML model if new data is available
            t0_all = time()
            t0_train = time()
            t1_train = time()

            # Generate new rattled + MLrelaxed candidate
            t_newCand_start = time()
            a_new, do_dp = self.newCandidate_beyes()
            t_newCand_end = time()
            
            # Singlepoint with objective potential
            t_sp_start = time()
            Enew, Fnew = self.singlePoint(a_new)
            t_sp_end = time()

            self.comm.barrier()
            print('{} '.format(self.comm.rank), end='')
            sys.stdout.flush()
            self.comm.barrier()
            if self.master:
                print('')
                sys.stdout.flush()
            self.comm.barrier()
                        
            if self.master:
                print('hello8')
                sys.stdout.flush()
            self.comm.barrier()

            # Try to add the new structure to the population
            t1_all = time()
            self.update_MLrelaxed_pop()

            if self.master:
                print('hello9')
                sys.stdout.flush()
            self.comm.barrier()
            
            self.population.add_structure(a_new, Enew, Fnew)

            if self.master:
                print('hello10')
                sys.stdout.flush()
            self.comm.barrier()
                    
    def update_MLrelaxed_pop(self):
        #  Initialize MLrelaxed population
        self.population.pop_MLrelaxed = []

        if self.master:
            print('hello81')
            sys.stdout.flush()
        self.comm.barrier()
        
        for a in self.population.pop:
            self.population.pop_MLrelaxed.append(a.copy())

        if self.master:
            print('hello82')
            sys.stdout.flush()
        self.comm.barrier()

        if self.comm.rank < len(self.population.pop):
            index = self.comm.rank
            #self.population.pop_MLrelaxed[index] = self.relaxML(self.population.pop[index], Fmax=0.01)
            self.population.pop_MLrelaxed[index] = self.population.pop[index]

        if self.master:
            print('hello83')
            sys.stdout.flush()
        self.comm.barrier()

        for i in range(len(self.population.pop)):
            pos = self.population.pop_MLrelaxed[i].positions
            self.comm.broadcast(pos, i)
            self.population.pop_MLrelaxed[i].set_positions(pos)
                
    def get_dualPoint(self, a, F, lmax=0.1, Fmax_flat=5):
        """
        lmax:
        The atom with the largest force will be displaced by this distance
        
        Fmax_flat:
        max displaced distance is increased linearely with force until 
        Fmax = Fmax_flat, over which it will be constant as lmax.
        """
        a_dp = a.copy()

        # Calculate and set new positions
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = lmax * F*min(1/Fmax_flat, 1/Fmax)
        pos_dp = a.positions + pos_displace
        a_dp.set_positions(pos_dp)
        return a_dp

    def get_force_mutated_population(self, lmax=[0.02, 0.07]):
        Npop = len(self.population.pop)
        Nl = len(lmax)
        E_list = np.zeros(Npop*Nl)
        error_list = np.zeros(Npop*Nl)
        pos_list = np.zeros((Npop*Nl, 3*self.Natoms))
        for i, a in enumerate(self.population.pop):
            F = a.get_forces()
            for n, l in enumerate(lmax):
                anew = self.get_dualPoint(a, F, lmax=l)
                pos_new = anew.get_positions()
                E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
                E_list[i*Nl+n] = E
                error_list[i*Nl+n] = error
                pos_list[i*Nl+n] = pos_new.reshape(-1)
        return pos_list, E_list, error_list
        
    def mutate(self, Ntasks_each):
        a_mutated_list = []
        for k in range(Ntasks_each):
            # draw random structure to mutate from population
            a = self.population.get_structure()
            a_copy = a.copy()
            a_mutated, _ = self.mutationSelector.get_new_individual([a_copy])
            a_mutated_list.append(a_mutated)
        self.comm.barrier()

        return a_mutated_list
    
    def newCandidate_beyes(self):
        anew = self.population.pop[-1].copy()
        pos_new = np.zeros((self.Natoms, 3))
        if self.master:
            pos_new = np.ones((self.Natoms, 3))
        self.comm.broadcast(pos_new, 0)
        anew.positions = pos_new
        self.comm.barrier()

        if self.master:
            print('hello5')
            sys.stdout.flush()
        self.comm.barrier()
        self.comm.barrier()
        
        self.traj_counter += 1
        return anew, False

    def trainModel(self):
        """
        # Reduce training data - If there is too much
        if self.ksaved > self.maxNtrain:
            Nremove = self.ksaved - self.maxNtrain
            self.ksaved = self.maxNtrain
            self.MLmodel.remove_data(Nremove)
        """
        #GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1, 3, 5)}
        GSkwargs = {'reg': [1e-5], 'sigma': [30]}
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
                
    def relaxML(self, anew, Fmax=0.1):
        a = anew.copy()

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        krr_calc = krr_calculator(self.MLmodel)
        a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=1, forcemax=Fmax)

        #self.traj_counter += 1
        return a_relaxed

    def relaxTrue(self, a):
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
        Frelaxed = a_relaxed.get_forces()
        return a_relaxed, Erelaxed, Frelaxed
            
    def singlePoint(self, anew):
        a = anew.copy()

        # broadcast structure, so all cores have the same
        E = np.zeros(1)
        if self.master:
            E = np.random.rand(1)
        self.comm.broadcast(E, 0)
        self.comm.barrier()

        F = np.zeros((self.Natoms, 3))
        if self.master:
            F = np.ones((self.Natoms, 3))
        self.comm.broadcast(F, 0)
        self.comm.barrier()

        # save structure for training
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
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.standardmutations_mkb import RattleMutation
    from ase.ga.data_esb import DataConnection
    from ase.ga.utilities import get_all_atom_types

    from prepare_startGenerator import prepare_startGenerator
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
    comparator = gaussComparator(featureCalculator=featureCalculator)
    delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator,
                    delta_function=delta_function)

    """
    # Set up operationSelector to handle mutations
    da = DataConnection('gadb.db')

    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=0.7)
    
    initial_structures = read('start_pop.traj', index=':2')
    """

    sg = prepare_startGenerator()
    atom_numbers_to_optimize = sg.atom_numbers
    n_to_optimize = len(atom_numbers_to_optimize)
    blmin = sg.blmin

    mutationSelector = OperationSelector([0.4, 0.3, 0.3],
                                         [sg,
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=0.7, rattle_prop=1.),
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=4, rattle_prop=0.2)])
    
    # Savefile setup
    savefiles_path = sys.argv[1]
    try:
        run_num = sys.argv[2]
    except IndexError:
        run_num = ''
    savefiles_namebase = savefiles_path + 'global' + run_num + '_' 

    optimizer = globalOptim(traj_namebase=savefiles_namebase,
                            MLmodel=krr,
                            a0=a, 
                            mutationSelector=mutationSelector,
                            Niter=10000,
                            dualPoint=True)

    optimizer.runOptimizer()
