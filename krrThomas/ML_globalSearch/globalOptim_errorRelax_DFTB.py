import numpy as np
from time import time

from custom_calculators import krr_calculator

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import BFGS, BFGSLineSearch
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga.relax_attaches import VariansBreak

from populationMC import population

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak

import os

import ase.parallel as mpi
world = mpi.world


def relaxGPAW(structure, label, calc=None, forcemax=0.1, niter_max=1, steps=10):

    # Create calculator
    if calc is None:
        calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),  # C
                  mode = 'lcao',
                  basis = 'dzp',
                  xc='PBE',
                  gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),  # C
                  occupations=FermiDirac(0.1),
                  maxiter=99,  # C
                  #maxiter=49,  # Sn3O3
                  mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
                  nbands=-50,
                  #kpts=(1,1,1),  # C
                  kpts=(2,2,1),  # Sn3O3
                  txt = label+ '_lcao.txt')
    else:
        calc.set(txt=label+'_true.txt')

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

def singleGPAW(structure, label, calc=None):
    # Create calculator
    if calc is None:
        calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),  # C
                  mode = 'lcao',
                  basis = 'dzp',
                  xc='PBE',
                  gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),  # C
                  occupations=FermiDirac(0.1),
                  maxiter=99,  # C
                  #maxiter=49,  # Sn3O3
                  mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
                  nbands=-50,
                  #kpts=(1,1,1),  # C
                  kpts=(2,2,1),  # Sn3O3
                  txt = label+ '_lcao.txt')
    else:
        calc.set(txt=label+'_true.txt')
    
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

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < niter_max:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 200)
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
    def __init__(self, traj_namebase, MLmodel, startGenerator, mutationSelector, calc=None, startStructures=None, population_size=5, kappa=2, Niter=50, Ninit=2, sigma=20, min_saveDifference=0.3, minSampleStep=10, dualPoint=False, relaxFinalPop=False, stat=True):

        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel
        self.startGenerator = startGenerator
        self.mutationSelector = mutationSelector
        self.calc = calc
        self.startStructures = startStructures
        
        self.population = population(population_size=population_size, comparator=self.MLmodel.comparator)
        
        self.kappa = kappa
        self.Natoms = len(self.startGenerator.slab) + len(self.startGenerator.atom_numbers)
        self.Niter = Niter
        self.Ninit = Ninit
        try:
            len(sigma)
            self.sigma = list(sigma)
        except:
            self.sigma = [sigma]
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.dualPoint = dualPoint
        self.relaxFinalPop = relaxFinalPop
        self.stat = stat

        self.opNameList = [op.descriptor for op in self.mutationSelector.oplist]
        
        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0
        
        # Define parallel communication
        self.comm = world.new_communicator(np.array(range(world.size)))
        self.master = self.comm.rank == 0

        # Make new folders
        self.ML_dir = 'all_MLcandidates/'
        self.pop_dir = 'relaxedPop/'
        if self.master:
            os.makedirs(self.ML_dir)
            os.makedirs(self.pop_dir)
        
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
                a_init, _ = self.startGenerator.get_new_individual()
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

            # Clean similar structures from population
            self.update_MLrelaxed_pop()
            
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
            #self.update_MLrelaxed_pop()
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
            self.traj_counter += 1
            
        # Save final population
        self.update_MLrelaxed_pop()
        pop = self.population.pop
        write(self.traj_namebase + 'finalPop.traj', pop)

        # relax final pop with true potential
        if self.relaxFinalPop:
            relaxed_pop = []
            for i, a in enumerate(pop):
                # Only promicing structures
                if (a.get_forces()**2).sum(axis = 1).max()**0.5 < 2:
                    name = savefiles_namebase + 'pop{}'.format(i)
                    a_relaxed = relaxGPAW(a, name, forcemax=0.05, steps=30, niter_max=2)
                    relaxed_pop.append(a_relaxed)

            write(self.traj_namebase + 'finalPop_relaxed.traj', relaxed_pop)
            
                    
    def update_MLrelaxed_pop(self):
        #  Initialize MLrelaxed population
        self.population.pop_MLrelaxed = []

        for a in self.population.pop:
            self.population.pop_MLrelaxed.append(a.copy())

        E_relaxed_pop = np.zeros(len(self.population.pop))
        error_relaxed_pop = np.zeros(len(self.population.pop))
        if self.comm.rank < len(self.population.pop):
            index = self.comm.rank
            a_MLrelaxed = self.relaxML(self.population.pop[index], Fmax=0.01)
            self.population.pop_MLrelaxed[index] = a_MLrelaxed
            E_temp, error_temp, _ = self.MLmodel.predict_energy(a_MLrelaxed, return_error=True)
            E_relaxed_pop[index] = E_temp
            error_relaxed_pop[index] = error_temp
            
        for i in range(len(self.population.pop)):
            pos = self.population.pop_MLrelaxed[i].positions
            self.comm.broadcast(pos, i)
            self.population.pop_MLrelaxed[i].set_positions(pos)

            E = np.array([E_relaxed_pop[i]])
            error = np.array([error_relaxed_pop[i]])
            self.comm.broadcast(E, i)
            self.comm.broadcast(error, i)
            
            self.population.pop_MLrelaxed[i].info['key_value_pairs']['predictedEnergy'] = E[0]
            self.population.pop_MLrelaxed[i].info['key_value_pairs']['predictedError'] = error[0]
            self.population.pop_MLrelaxed[i].info['key_value_pairs']['fitness'] = E[0] - self.kappa*error[0]

        label = self.pop_dir + 'ML_relaxed_pop{}'.format(self.traj_counter)
        write(label+'.traj', self.population.pop_MLrelaxed)
        
        """
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
        """
            
    def get_dualPoint(self, a, F, lmax=0.10, Fmax_flat=5):
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
        if self.master:
            t0 = time()
        anew_mutated_list = self.mutate(N_tasks)
        if self.master:
            print('mutation time:', time() - t0, flush=True)
        
        # Relax with MLmodel
        anew_list = []
        E_list = []
        error_list = []
        for anew_mutated in anew_mutated_list:
            anew = self.relaxML(anew_mutated, with_error=True)
            anew_list.append(anew)
            
            E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
            E_list.append(E)
            error_list.append(error)
        E_list = np.array(E_list)
        error_list = np.array(error_list)

        if self.master:
            print('theta0:', theta0, flush=True)
        
        oplist_index = np.array([self.opNameList.index(a.info['key_value_pairs']['origin']) for a in anew_list]).astype(int)
        
        # Gather data from slaves to master
        pos_new_list = np.array([anew.positions for anew in anew_list])
        pos_new_mutated_list = np.array([anew_mutated.positions for anew_mutated in anew_mutated_list])
        if self.comm.rank == 0:
            E_all = np.empty(N_tasks * self.comm.size, dtype=float)
            error_all = np.empty(N_tasks * self.comm.size, dtype=float)
            pos_all = np.empty(N_tasks * 3*self.Natoms*self.comm.size, dtype=float)
            pos_all_mutated = np.empty(N_tasks * 3*self.Natoms*self.comm.size, dtype=float)
            oplist_index_all = np.empty(N_tasks * self.comm.size, dtype=int)
        else:
            E_all = None
            error_all = None
            pos_all = None
            pos_all_mutated = None
            oplist_index_all = None
        self.comm.gather(E_list, 0, E_all)
        self.comm.gather(error_list, 0, error_all)
        self.comm.gather(pos_new_list.reshape(-1), 0, pos_all)
        self.comm.gather(pos_new_mutated_list.reshape(-1), 0, pos_all_mutated)
        self.comm.gather(oplist_index, 0, oplist_index_all)
        
        # Pick best candidate on master + broadcast
        pos_new = np.zeros((self.Natoms, 3))
        pos_new_mutated = np.zeros((self.Natoms, 3))
        if self.master:
            fitness_all = E_all - self.kappa * error_all
            index_best = fitness_all.argmin()
            
            print('{}:\n'.format(self.traj_counter), np.c_[E_all, error_all])
            print('{} best:\n'.format(self.traj_counter), E_all[index_best], error_all[index_best])
        
            with open(self.traj_namebase + 'E_MLerror.txt', 'a') as f:
                f.write('{0:.4f}\t{1:.4f}\n'.format(E_all[index_best], error_all[index_best]))
            
            pos_all = pos_all.reshape((-1, self.Natoms, 3))
            pos_new = pos_all[index_best]  # old
            pos_all_mutated = pos_all_mutated.reshape((N_tasks * self.comm.size, self.Natoms, 3))
            pos_new_mutated = pos_all_mutated[index_best]  # old
            
            if self.stat:
                a_all = []
                a_all_mutated = []
                for i, (pos, pos_mutated) in enumerate(zip(pos_all, pos_all_mutated)):
                    a = anew.copy()
                    a_mutated = anew.copy()
                    a.positions = pos
                    a_mutated.positions = pos_mutated
                    a.info['key_value_pairs']['predictedEnergy'] = E_all[i]
                    a.info['key_value_pairs']['predictedError'] = error_all[i]
                    a.info['key_value_pairs']['fitness'] = E_all[i] - self.kappa*error_all[i]
                    a.info['key_value_pairs']['origin'] = self.opNameList[oplist_index_all[index_best]]

                    a_all.append(a)
                    a_all_mutated.append(a_mutated)
                    
                label_relaxed = self.ML_dir + 'ML_relaxed{}'.format(self.traj_counter)
                write(label_relaxed+'.traj', a_all, parallel=False)
                label_unrelaxed = self.ML_dir + 'ML_unrelaxed{}'.format(self.traj_counter)
                write(label_unrelaxed+'.traj', a_all_mutated, parallel=False)

            # Write unrelaxed + relaxed versions of best candidate to file
            label = self.ML_dir + 'ML_best{}'.format(self.traj_counter)
            write(label+'.traj', [a_all_mutated[index_best], a_all[index_best]], parallel=False)
            
            #try:
            #    pos_new = pos_all[index_best]
            #    pos_new_mutated = pos_all_mutated[index_best]
            #except IndexError:
            #    pos_new = self.population.pop_MLrelaxed[index_best % N_newCandidates].get_positions()
            #    pos_new_mutated = self.population.pop[index_best % N_newCandidates].get_positions()
                
            

        self.comm.broadcast(pos_new, 0)
        anew.positions = pos_new
        self.comm.broadcast(pos_new_mutated, 0)
        anew_mutated = anew.copy()
        anew_mutated.positions = pos_new_mutated
        self.comm.barrier()

        # Write unrelaxed + relaxed versions of best candidate to file
        #label = self.ML_dir + 'ML_best{}'.format(self.traj_counter)
        #write(label+'.traj', [anew_mutated, anew])

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
        GSkwargs = {'reg': [1e-5], 'sigma': self.sigma}  # SnO
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         k=3,
                                         **GSkwargs)

        self.a_add = []
        if self.master:
            with open(self.traj_namebase + 'sigma.txt', 'a') as f:
                f.write('{0:.2f}\n'.format(params['sigma']))

    def add_trajectory_to_training(self, trajectory_file):
        atoms = read(filename=trajectory_file, index=':', parallel=False)
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
                
    def relaxML(self, anew, Fmax=0.1, with_error=True):
        a = anew.copy()

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        if with_error:
            krr_calc = krr_calculator(self.MLmodel, kappa=self.kappa)
        else:
            krr_calc = krr_calculator(self.MLmodel)
        a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=1, forcemax=Fmax)

        return a_relaxed

    def relaxTrue(self, a):
        pos = a.positions
        if self.master:
            pos = a.positions
        self.comm.broadcast(pos, 0)
        a.positions = pos
        self.comm.barrier()

        label = self.traj_namebase + '{}'.format(self.traj_counter)
        
        pos_relaxed = a.positions
        Erelaxed = np.zeros(1)
        Frelaxed = np.zeros((3,self.Natoms))
        if self.master:
            # Relax
            a_relaxed = relaxGPAW(a, label, calc=self.calc)
            pos_relaxed = a_relaxed.positions
            Erelaxed = np.array([a_relaxed.get_potential_energy()])
            Frelaxed = a_relaxed.get_forces()
        self.comm.broadcast(pos_relaxed, 0)
        self.comm.broadcast(Erelaxed, 0)
        Erelaxed = Erelaxed[0]
        self.comm.broadcast(Frelaxed, 0)
        a_relaxed = a.copy()
        a_relaxed.positions = pos_relaxed        
        self.comm.barrier()

        # Add sampled trajectory to training data.
        self.add_trajectory_to_training(label+'_lcao.traj')

        
        self.traj_counter += 1
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
        E = np.zeros(1)
        F = np.zeros((self.Natoms,3))
        if self.master:
            label =  self.traj_namebase + '{}'.format(self.traj_counter)
            E, F = singleGPAW(a, label, calc=self.calc)
            E = np.array([E])
        self.comm.broadcast(E, 0)
        E = E[0]
        self.comm.broadcast(F, 0)
        
        
        # save structure for training
        a.energy = E
        results = {'energy': E}
        calc = SinglePointCalculator(a, **results)
        a.set_calculator(calc)
        self.a_add.append(a)

        # Save to spTrain-trajectory
        self.writer_spTrain.write(a, energy=E, forces=F)
        self.ksaved += 1
        return E, F

if __name__ == '__main__':
    from gaussComparator import gaussComparator
    from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
    from delta_functions_multi.delta import delta as deltaFunc
    from krr_ase import krr_class
    
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.standardmutations_mkb import RattleMutation, PermutationMutation

    from prepare_startGenerator import prepare_startGenerator

    import sys
    
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
    eta = 20
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
                            Niter=500,
                            Ninit=2,
                            dualPoint=True)

    optimizer.runOptimizer()

    
    
    
