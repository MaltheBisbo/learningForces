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
from ML_globalSearch.parallel_utilities import sync_atoms

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak

import traceback
import sys
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
        dyn = BFGSLineSearch(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 200)
        niter += 1
        
    return structure

class globalOptim_baseClass():
    """
    --Input--
    MLmodel:
    Model that given training data can predict energy and gradient of a structure.
    Hint: must include a fit, predict_energy and predict_force methods.
    
    Natoms:
    Number of atoms in structure.
    
    Niter:
    Number of monte-carlo steps.

    """
    def __init__(self, traj_namebase, MLmodel, startGenerator, mutationSelector, calc=None, startStructures=None, population_size=5, kappa=2, Niter=50, Ninit=2, sigma=20, reg=1e-5, Ncand_min=30, min_certainty=0.8, dualPoint=False, errorRelax=False, relaxFinalPop=False):

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
        self.reg = reg
        self.Ncand_min = Ncand_min
        self.min_certainty = min_certainty
        self.dualPoint = dualPoint
        self.errorRelax = errorRelax
        self.relaxFinalPop = relaxFinalPop
        
        self.operation_dict = [op.descriptor for op in self.mutationSelector.oplist]
        
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

            sp_successfull = False
            for index_surogate_search in range(5):
                # Generate new rattled + MLrelaxed candidate
                t_newCand_start = time()
                a_all, a_mutated_all, E_all, error_all = self.newCandidate_beyes()
                t_newCand_end = time()
                
                # Singlepoint on best
                t_sp_start = time()
                kappa0 = self.kappa
                for index_sp in range(5):
                    index_best = np.argmin(E_all - kappa0*error_all)
                    a_new = a_all[index_best]
                    try:
                        Enew, Fnew = self.singlePoint(a_new)
                        sp_successfull = True
                        break
                    except Exception as runtime_err:
                        kappa0 /= 2
                        if self.master:
                            print('Error in SP-attempt {} cought:'.format(index_sp), runtime_err, flush=True)
                            traceback.print_exc()
                        traceback.print_exc(file=sys.stderr)
                            
                t_sp_end = time()
                if sp_successfull:
                    break
                else:
                    if self.master:
                        print('all SP-attempts failed for candidate-set no.', index_surogate_search)
            
            if self.master:
                print('{} best:\n'.format(self.traj_counter), E_all[index_best], error_all[index_best])
                label_best = self.ML_dir + 'ML_best{}'.format(self.traj_counter)
                write(label_best+'.traj', [a_mutated_all[index_best], a_new], parallel=False)

            # Get dualpoint
            a_dp = self.get_dualPoint(a_new, Fnew)
            try:
                E_dp, F_dp = self.singlePoint(a_dp)
                if E_dp < Enew:
                    a_new = a_dp.copy()
                    Enew = E_dp
                    Fnew = F_dp
                if self.master:
                    write(label_best+'.traj', a_dp, parallel=False, append=True)
            except Exception as runtime_err:
                if self.master:
                    print('Error in dualPoint cought:', runtime_err, flush=True)
                    traceback.print_exc()
                traceback.print_exc(file=sys.stderr)
                    
            # Try to add the new structure to the population
            t1_all = time()
            #self.update_MLrelaxed_pop()
            Ebest = self.population.pop[0].get_potential_energy()
            if sp_successfull and Enew < Ebest+30:
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
                    f.write('newCand:{0:.2f}\tsp:{1:.2f}\ttrain:{2:.2f}\twdp:{3:.2f}\tall:{4:.2f}\n'.format(t_newCand_end-t_newCand_start,
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
            a_MLrelaxed = self.relaxML(self.population.pop[index], Fmax=0.005)
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

        label_MLrelaxed_pop = self.pop_dir + 'ML_relaxed_pop{}'.format(self.traj_counter)
        write(label_MLrelaxed_pop+'.traj', self.population.pop_MLrelaxed)
        label_pop = self.pop_dir + 'pop{}'.format(self.traj_counter)
        write(label_pop+'.traj', self.population.pop)
        
        self.comm.barrier()
        if self.master:
            print('Relaxing population done', flush=True)
            
    def get_forcePerturbed(self, a, F, lmax=0.10):
        """
        lmax:
        The atom with the largest force will be displaced by this distance
        """
        anew = a.copy()

        # Calculate and set new positions
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = lmax * F/Fmax
        pos_new = a.positions + pos_displace
        anew.set_positions(pos_new)
        return anew
        
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

    def get_force_mutated_population(self, lmax=[0.05, 0.1, 0.2]):
        Npop = len(self.population.pop)
        Nl = len(lmax)
        pop_preMut = []
        pop_forceMut = []
        for a in self.population.pop:
            for _ in range(Nl):
                pop_forceMut.append(a.copy())
                a_cp = a.copy()
                a_cp.info['key_value_pairs']['origin'] = 'ForceMutation'
                pop_preMut.append(a_cp)

        self.comm.barrier()
        if self.master:
            print('force mutation part 1 done', flush=True)
        
        E_list = np.zeros(Npop*Nl)
        error_list = np.zeros(Npop*Nl)
        pos_list = np.zeros((Npop*Nl, 3*self.Natoms))
        if self.comm.rank < Npop:
            i = self.comm.rank
            F = self.population.pop[i].get_forces()
            for n, l in enumerate(lmax):
                index = Nl*i+n
                anew = self.get_forcePerturbed(pop_forceMut[index], F, lmax=l)
                pop_forceMut[index] = anew
                E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
                E_list[index] = E
                error_list[index] = error

        self.comm.barrier()
        if self.master:
            print('force mutation part 2 done', flush=True)

        for i in range(Npop):
            for n in range(Nl):
                index = Nl*i+n
                pos = pop_forceMut[index].get_positions()
                self.comm.broadcast(pos, i)
                pop_forceMut[index].set_positions(pos)

                E = np.array([E_list[index]])
                error = np.array([error_list[index]])
                self.comm.broadcast(E, i)
                self.comm.broadcast(error, i)
                
                pop_forceMut[index].info['key_value_pairs']['predictedEnergy'] = E[0]
                pop_forceMut[index].info['key_value_pairs']['predictedError'] = error[0]
                pop_forceMut[index].info['key_value_pairs']['fitness'] = E[0] - self.kappa*error[0]
                pop_forceMut[index].info['key_value_pairs']['origin'] = 'ForceMutation'

        self.comm.barrier()
        if self.master:
            print('force mutated candidates done', flush=True)
                
        return pop_forceMut, pop_preMut

    def mutate(self, Ntasks_each):
        Ntrials = 5
        a_mutated_list = []
        for k in range(Ntasks_each):
            # Draw random structure to mutate from population
            # and use to generate new candidate.
            for i_trial in range(Ntrials):
                parents = self.population.get_structure_pair()
                a_mutated, _ = self.mutationSelector.get_new_individual(parents)
                # break trial loop if successful
                if a_mutated is not None:
                    a_mutated_list.append(a_mutated)
                    break
            # If no success in max number of trials
            if a_mutated is None:
                a_mutated = parents[0].copy()
                a_mutated_list.append(a_mutated)
        self.comm.barrier()

        return a_mutated_list

    def newCandidate_beyes(self):
        N_newCandidates = self.Ncand_min

        # Number of new candidated made by each core
        Neach = int(np.ceil(N_newCandidates / self.comm.size))

        # Use all cores on nodes.
        N_newCandidates = Neach * N_newCandidates
        
        # perform mutations
        if self.master:
            t0 = time()
        anew_mutated_list = self.mutate(Neach)
        if self.master:
            print('mutation time:', time() - t0, flush=True)
        
        # Relax with MLmodel
        anew_list = []
        E_list = []
        error_list = []
        for anew_mutated in anew_mutated_list:
            anew = self.relaxML(anew_mutated, with_error=self.errorRelax)
            anew_list.append(anew)
            E, error, theta0 = self.MLmodel.predict_energy(anew, return_error=True)
            E_list.append(E)
            error_list.append(error)
        E_list = np.array(E_list)
        error_list = np.array(error_list)

        if self.master:
            print('theta0:', theta0, flush=True)
        
        operation_index = np.array([self.operation_dict.index(a.info['key_value_pairs']['origin'])
                                    for a in anew_list]).astype(int)

        # Syncronize all new candidates on all cores
        anew_mutated_all = sync_atoms(world, atoms_list=anew_mutated_list,
                                   operation_dict=self.operation_dict, operation_index=operation_index)
        anew_all = sync_atoms(world, atoms_list=anew_list, Epred_list=E_list,
                              error_list=error_list, kappa=self.kappa,
                              operation_dict=self.operation_dict, operation_index=operation_index)

        
        error_all_tmp = np.array([a.info['key_value_pairs']['predictedError'] for a in anew_all])
        min_certainty = self.min_certainty
        for _ in range(5):
            filt = error_all_tmp < min_certainty*np.sqrt(np.abs(theta0))
            if np.sum(filt.astype(int)) > 0:
                anew_mutated_all = [anew_mutated_all[i] for i in range(len(anew_all)) if filt[i]]
                anew_all = [anew_all[i] for i in range(len(anew_all)) if filt[i]]
                break
            else:
                min_certainty = min_certainty + (1-min_certainty)/2

        self.comm.barrier()
        if self.master:
            print('model relaxed candidates done', flush=True)
        
        # Write candidates to file
        if self.master:
            label_relaxed = self.ML_dir + 'ML_relaxed{}'.format(self.traj_counter)
            write(label_relaxed+'.traj', anew_all, parallel=False)
            label_unrelaxed = self.ML_dir + 'ML_unrelaxed{}'.format(self.traj_counter)
            write(label_unrelaxed+'.traj', anew_mutated_all, parallel=False)

        # Add force-mutated structures to candidates
        #anew_forceMut, anew_preForceMut = self.get_force_mutated_population()
        #anew_all += anew_forceMut
        #anew_mutated_all += anew_preForceMut
            
        # Extract + print data
        E_all = np.array([a.info['key_value_pairs']['predictedEnergy'] for a in anew_all])
        error_all = np.array([a.info['key_value_pairs']['predictedError'] for a in anew_all])
        fitness_all = np.array([a.info['key_value_pairs']['fitness'] for a in anew_all])
        if self.master:
            print('{}:\n'.format(self.traj_counter), np.c_[E_all, error_all, fitness_all])
            
        return anew_all, anew_mutated_all, E_all, error_all 

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
        GSkwargs = {'reg': [self.reg], 'sigma': self.sigma}  # SnO
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         k=3,
                                         **GSkwargs)

        self.a_add = []
        if self.master:
            with open(self.traj_namebase + 'sigma.txt', 'a') as f:
                f.write('{0:.2f}\n'.format(params['sigma']))

        self.comm.barrier()
        if self.master:
            print('Training done', flush=True)
                
    def add_trajectory_to_training(self, trajectory_file):
        traj = read(filename=trajectory_file, index=':', parallel=False)
        E = [a.get_potential_energy() for a in traj]
        N = len(traj)

        index_save = np.arange(N-1, -1, -3)
        for i in index_save:
            self.a_add.append(traj[i])
            self.writer_initTrain.write(traj[i], energy=E[i])

    def relaxML(self, anew, Fmax=0.1, with_error=False):
        a = anew.copy()

        # Relax
        label = self.traj_namebase + 'ML{}'.format(self.traj_counter)
        if with_error:
            krr_calc = krr_calculator(self.MLmodel, kappa=self.kappa)
        else:
            krr_calc = krr_calculator(self.MLmodel)
        a_relaxed = relax_VarianceBreak(a, krr_calc, label, niter_max=1, forcemax=Fmax)

        return a_relaxed

    
    
    
