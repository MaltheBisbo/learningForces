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
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),  # C
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),  # C
              occupations=FermiDirac(0.1),
              #maxiter=99,  # C
              maxiter=49,  # Sn3O3
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              #kpts=(1,1,1),  # C
              kpts=(2,2,1),  # Sn3O3
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

    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),  # C
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),  # C
              occupations=FermiDirac(0.1),
              #maxiter=99,  # C
              maxiter=49,  # Sn3O3
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              #kpts=(1,1,1),  # C
              kpts=(2,2,1),  # Sn3O3
              txt = label+ '_lcao.txt')
    
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
    def __init__(self, traj_namebase, MLmodel, startGenerator, mutationSelector, startStructures=None, population_size=5, kappa=2, Niter=50, Ninit=2, min_saveDifference=0.3, minSampleStep=10, dualPoint=False):

        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel
        self.startGenerator = startGenerator
        self.mutationSelector = mutationSelector
        self.startStructures = startStructures
        
        self.population = population(population_size=population_size, comparator=self.MLmodel.comparator)
        
        self.kappa = kappa
        self.Natoms = len(self.startGenerator.slab) + len(self.startGenerator.atom_numbers)
        self.Niter = Niter
        self.Ninit = Ninit
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
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
        if self.startStructures is None:
            for i in range(self.Ninit):
                a_init = self.startGenerator.get_new_candidate()
                a, E, F = self.relaxTrue(a_init)
                self.population.add_structure(a, E, F)
        else:
            for a in self.startStructures:
                Ei = a.get_potential_energy()
                Fi = a.get_forces()
                self.a_add.append(a)
                self.writer_initTrain.write(a, energy=Ei)
                self.population.add_structure(a, Ei, Fi)
        
        # Reset traj_counter for ML-relaxations
        self.traj_counter = 0
        
        # Run global search
        for i in range(self.Niter):
            # Train ML model if new data is available
            t0_all = time()
            t0_train = time()
            if len(self.a_add) > 0:
                self.trainModel()
            t1_train = time()

            # Generate new rattled + MLrelaxed candidate
            t_newCand_start = time()
            a_new = self.newCandidate_beyes()
            t_newCand_end = time()

            # Singlepoint with objective potential
            t_sp_start = time()
            Enew, Fnew = self.singlePoint(a_new)
            t_sp_end = time()
            
            # Get dual-point if relevant
            Fnew_max = (Fnew**2).sum(axis=1).max()**0.5
            if self.dualPoint and i > 50 and Fnew_max > 0.5:
                a_dp = self.get_dualPoint(a_new, Fnew)
                E, error, _ = self.MLmodel.predict_energy(a_dp, return_error=True)
                # If dual-point looks promising - perform sp-calculation
                if E - self.kappa*error < self.population.largest_energy: 
                    E_dp, F_dp = self.singlePoint(a_dp)
                    if E_dp < Enew:
                        a_new = a_dp.copy()
                        Enew = E_dp
                        Fnew = F_dp

            # Try to add the new structure to the population
            t1_all = time()
            self.update_MLrelaxed_pop()
            self.population.add_structure(a_new, Enew, Fnew)
            
            if self.master:
                for i, a in enumerate(self.population.pop):
                    E = a.get_potential_energy()
                    print('pop{0:d}={1:.2f}  '.format(i, E), end='')
                    
                    # write population to file
                    self.writer_current.write(a, energy=E, forces=a.get_forces())
                print('')
                print('Enew={}'.format(Enew))
            t2_all = time()
            if self.master:
                with open(self.traj_namebase + 'time.txt', 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(t_newCand_end-t_newCand_start,
                                                          t_sp_end-t_sp_start,
                                                          t1_train - t0_train,
                                                          t1_all - t0_all,
                                                          t2_all - t0_all))
                    
    def update_MLrelaxed_pop(self):
        #  Initialize MLrelaxed population
        self.population.pop_MLrelaxed = []

        for a in self.population.pop:
            self.population.pop_MLrelaxed.append(a.copy())

        if self.comm.rank < len(self.population.pop):
            index = self.comm.rank
            self.population.pop_MLrelaxed[index] = self.relaxML(self.population.pop[index], Fmax=0.01)

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
        N_newCandidates = 30

        # the maximum number of candidates a core need to make N_newCandidates on a single node.
        N_tasks = int(np.ceil(N_newCandidates / self.comm.size))

        # Use all cores on nodes.
        N_newCandidates = N_tasks * N_newCandidates

        # perform mutations
        anew_mutated_list = self.mutate(N_tasks)
        
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
        
        # Pick best candidate on master + broadcast
        pos_new = np.zeros((self.Natoms, 3))
        pos_new_mutated = np.zeros((self.Natoms, 3))
        if self.master:
            
            EwithError_all = E_all - self.kappa * error_all
            index_best = EwithError_all.argmin()
            
            print('{}:\n'.format(self.traj_counter), np.c_[E_all, error_all])
            print('{} best:\n'.format(self.traj_counter), E_all[index_best], error_all[index_best])
            
            with open(self.traj_namebase + 'E_MLerror.txt', 'a') as f:
                f.write('{0:.4f}\t{1:.4f}\n'.format(E_all[index_best], error_all[index_best]))
            
            pos_all = pos_all.reshape((-1, self.Natoms, 3))
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

    def trainModel(self):
        """
        # Reduce training data - If there is too much
        if self.ksaved > self.maxNtrain:
            Nremove = self.ksaved - self.maxNtrain
            self.ksaved = self.maxNtrain
            self.MLmodel.remove_data(Nremove)
        """
        #GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1, 3, 5)}
        # GSkwargs = {'reg': [1e-5], 'sigma': [30]}  # C24
        GSkwargs = {'reg': [1e-5], 'sigma': [15]}  # SnO

        if self.master:
            print('b1')
            sys.stdout.flush()
        self.comm.barrier()
        
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         k=5,
                                         **GSkwargs)

        if self.master:
            print('b2')
            sys.stdout.flush()
        self.comm.barrier()

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
        
        # Save structure with ML-energy
        if self.master:
            self.writer_spPredict.write(a)

        # broadcast structure, so all cores have the same
        pos = a.positions
        if self.master:
            pos = a.positions
        self.comm.broadcast(pos, 0)
        a.positions = pos
        self.comm.barrier()

        # Perform single-point
        label =  self.traj_namebase + '{}'.format(self.traj_counter)
        E, F = singleGPAW(a, label)
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
    #from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
    from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
    #from delta_functions.delta import delta as deltaFunc
    from delta_functions_multi.delta import delta as deltaFunc
    from krr_ase import krr_class
    
    from ase.calculators.dftb import Dftb
    import sys

    from ase.data import covalent_radii
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.standardmutations_mkb import RattleMutation, PermutationMutation
    from ase.ga.data_esb import DataConnection
    from ase.ga.utilities import get_all_atom_types

    from prepare_startGenerator import prepare_startGenerator

    # Set up startGenerator and mutations
    sg = prepare_startGenerator()
    atom_numbers_to_optimize = sg.atom_numbers
    n_to_optimize = len(atom_numbers_to_optimize)
    blmin = sg.blmin

    """
    mutationSelector = OperationSelector([0.3, 0.2, 0.2, 0.3],
                                         [sg,
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=0.3, rattle_prop=1.),
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=0.7, rattle_prop=1.),
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=2, rattle_prop=0.1)])
    """
    mutationSelector = OperationSelector([0.3, 0.3, 0.2, 0.2],
                                         [sg,
                                          PermutationMutation(n_to_optimize, probability=0.3),
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=0.7, rattle_prop=1.),
                                          RattleMutation(blmin, n_to_optimize,
                                                         rattle_strength=2, rattle_prop=0.4)])

    a = sg.get_new_candidate()
    
    Rc1 = 6
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 2
    eta = 5
    use_angular = True
    
    """
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 5
    use_angular = True
    """
    
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    # Set up KRR-model
    comparator = gaussComparator(featureCalculator=featureCalculator)
    #delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
    delta_function = deltaFunc(atoms=a, rcut=6)
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator,
                    delta_function=delta_function)

    
    # Savefile setup
    savefiles_path = sys.argv[1]
    try:
        run_num = sys.argv[2]
    except IndexError:
        run_num = ''
    savefiles_namebase = savefiles_path + 'global' + run_num + '_' 

    try:
        start_pop_file = sys.argv[3]
        start_pop = read(start_pop_file, index=':')
    except IndexError:
        start_pop = None
    
    optimizer = globalOptim(traj_namebase=savefiles_namebase,
                            MLmodel=krr,
                            startGenerator=sg,
                            mutationSelector=mutationSelector,
                            startStructures=start_pop,
                            kappa=2,
                            Niter=300,
                            Ninit=2,
                            dualPoint=True)

    optimizer.runOptimizer()

    
    # Save final population
    optimizer.update_MLrelaxed_pop()
    pop = optimizer.population.pop
    write(savefiles_namebase + 'finalPop.traj', pop)

    
    relaxed_pop = []
    for i, a in enumerate(pop):
        name = savefiles_namebase + 'pop{}'.format(i)
        a_relaxed = relaxGPAW(a, name, forcemax=0.05, steps=200, niter_max=10)
        relaxed_pop.append(a_relaxed)

    write(savefiles_namebase + 'finalPop_relaxed.traj', relaxed_pop)
    
