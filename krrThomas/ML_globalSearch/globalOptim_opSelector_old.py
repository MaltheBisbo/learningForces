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
    
    Nstag:
    Max number of iterations without accepting new structure before
    the search is terminated.

    min_saveDifference:
    Defines the energy which a new trajectory point has to be lover than the previous, to be saved for training.
    
    minSampleStep:
    if minSampleStep=10, every tenth point in the relaxation trajectory is used for training, and so on..
    Unless min_saveDifference have not been ecxeeded.

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
    def __init__(self, calculator, traj_namebase, MLmodel, mutationSelector, population_size=5, kappa=2, Natoms=6, Niter=50, rattle_maxDist=0.5, Nstag=10, min_saveDifference=0.3, minSampleStep=10, Nstart_pop=5, Nperturb=2, noZ=False, dualPoint=False):

        self.calculator = calculator
        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel

        self.kappa = kappa

        self.Natoms = Natoms
        self.rattle_maxDist = rattle_maxDist
        self.Niter = Niter
        self.Nstag = Nstag
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.Nstart_pop = Nstart_pop
        self.Nperturb = Nperturb
        self.noZ = noZ
        self.dualPoint = dualPoint

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0

        self.population = population(population_size=population_size, comparator=self.MLmodel.comparator)
        
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
        for i in range(self.Nstart_pop):
            if self.noZ:
                a_init = createInitalStructure2d(self.Natoms)
            else:
                a_init = createInitalStructure(self.Natoms)
            a, E, F = self.relaxTrue(a_init)
            self.population.add_structure(a, E, F)
                
        # Reset traj_counter for ML-relaxations
        self.traj_counter = 0

        # Run global search
        stagnation_counter = 0
        for i in range(self.Niter):
            # Train ML model if new data is available
            if len(self.a_add) > 0:
                self.trainModel()
                
            # Generate new rattled + MLrelaxed candidate
            t_newCand_start = time()
            a_new = self.newCandidate_beyes()
            t_newCand_end = time()
            
            # Singlepoint with objective potential
            t_sp_start = time()
            Enew, Fnew = self.singlePoint(a_new)
            t_sp_end = time()
            if self.master:
                with open(self.traj_namebase + 'time.txt', 'a') as f:
                    f.write('{}\t{}\n'.format(t_newCand_end-t_newCand_start, t_sp_end-t_sp_start))
                    
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
                
    def get_dualPoint(self, a, F):
        max_atom_displacement = 0.1  # The atom with the largest force will be displaced this ammount
        Fmax_flat = 5
        a_dp = a.copy()

        # Calculate and set new positions
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = max_atom_displacement * F*min(1/Fmax_flat, 1/Fmax)
        pos_dp = a.positions + pos_displace
        a_dp.set_positions(pos_dp)
        return a_dp
        
    def mutate(self, Ntasks_each):
        N_newCandidates = self.comm.size * Ntasks_each
        
        N_exploit_rattle = int(0.3 * N_newCandidates)
        N_exploit_disk = int(0.3 * N_newCandidates)
        N_explore = N_newCandidates - N_exploit_rattle - N_exploit_disk

        mutation_tasks = [0]*3
        for i in range(N_newCandidates):
            rank = i % self.comm.size
            if self.comm.rank == rank:
                if i < N_explore:
                    mutation_tasks[0] += 1
                elif i < N_explore + N_exploit_disk:
                    mutation_tasks[1] += 1
                else:
                    mutation_tasks[2] += 1
        self.comm.barrier()
        
        a_mutated_list = []
        for i_mut, n_mut in enumerate(mutation_tasks):
            for k in range(n_mut):
                # draw random structure to mutate from population
                a = self.population.get_structure()
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
                        rmax_rattle = 0.1 + 0.7*np.random.rand()
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
                        rmax_rattle = 0.1 + 0.7*np.random.rand()
                        a_mutated = rattle_Natoms(struct=a_mutated,
                                                    Nrattle=self.Natoms,
                                                    rmax_rattle=rmax_rattle)
                a_mutated_list.append(a_mutated)
        self.comm.barrier()

        return a_mutated_list
    
    def newCandidate_beyes(self):
        N_newCandidates = 30

        # Number of candidates to be relaxed on each core
        N_tasks = int(np.ceil(N_newCandidates / self.comm.size))

        # Use all cores on nodes.
        N_newCandidates = N_tasks * N_newCandidates

        # Generate and destribute task list
        tasks = np.zeros(N_tasks).astype(int)
        if self.master:
            current_pop_size = len(self.population.pop)
            task_list = np.array([0]*current_pop_size + [1]*(N_newCandidates - current_pop_size))
        else:
            task_list = None
        self.comm.scatter(task_list, tasks, 0)


        
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
        # fix atoms in xy-plane if noZ=True
        if self.noZ:
            plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
            a.set_constraint(plane)

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        krr_calc = krr_calculator(self.MLmodel)
        a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=1, forcemax=Fmax)

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
    from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
    from krr_ase import krr_class
    #from krr_fracTrain import krr_class
    from delta_functions.delta import delta as deltaFunc
    from ase.calculators.dftb import Dftb

    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.standardmutations import RattleMutation
    from ase.ga.data_esb import DataConnection
    from ase.ga.utilities import get_all_atom_types
    
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
    comparator = gaussComparator(featureCalculator=featureCalculator)
    delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator,
                    delta_function=delta_function)


    # Set up operationSelector to handle mutations
    da = DataConnection('gadb.db')

    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=0.7)

    mutationSelector = OperationSelector([1.],
                                  [RattleMutation(blmin, n_to_optimize)])
    sdadadasdsad
    
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
                            mutationSelector=mutationSelector,
                            Natoms=Natoms,
                            Niter=600,
                            Nstag=600,
                            Nstart_pop=2,
                            Nperturb=2,
                            rattle_maxDist=5,
                            noZ=False,
                            dualPoint=True)

    optimizer.runOptimizer()
