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

from globalOptim_baseClass_dualGPR import relaxGPAW, singleGPAW, relax_VarianceBreak, globalOptim_baseClass

import os

import ase.parallel as mpi
world = mpi.world


class globalOptim(globalOptim_baseClass):

    def __init__(self, traj_namebase, MLmodel_prior, MLmodel_fine, startGenerator, mutationSelector, calc=None, startStructures=None, population_size=5, kappa=2, Niter=50, Ninit=2, Ncand_min=30, min_certainty=0.8, min_distance=0.0, dEmin_fine=30, Estd_threshold=10, Nrunning_mean=30, dualPoint=False, errorRelax=False, relaxFinalPop=False):

        globalOptim_baseClass.__init__(self,
                                       traj_namebase=traj_namebase,
                                       MLmodel_prior=MLmodel_prior,
                                       MLmodel_fine=MLmodel_fine,
                                       startGenerator=startGenerator,
                                       mutationSelector=mutationSelector,
                                       calc=calc,
                                       startStructures=startStructures,
                                       population_size=population_size,
                                       kappa=kappa,
                                       Niter=Niter,
                                       Ninit=Ninit,
                                       Ncand_min=Ncand_min,
                                       min_certainty=min_certainty,
                                       min_distance=min_distance,
                                       dEmin_fine=dEmin_fine,
                                       Estd_threshold=Estd_threshold,
                                       Nrunning_mean=Nrunning_mean,
                                       dualPoint=dualPoint,
                                       errorRelax=errorRelax,
                                       relaxFinalPop=relaxFinalPop)

    def relaxTrue(self, a):
        pos = a.positions
        if self.master:
            pos = a.positions
        self.comm.broadcast(pos, 0)
        a.positions = pos
        self.comm.barrier()

        # Relax
        label = self.traj_namebase + '{}'.format(self.traj_counter)
        a_relaxed = relaxGPAW(a, label, calc=self.calc)

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
        E, F = singleGPAW(a, label, calc=self.calc)
        self.comm.barrier()

        # save structure for training
        results = {'energy': E, 'forces': F}
        calc_sp = SinglePointCalculator(a, **results)
        a.set_calculator(calc_sp)
        self.a_add.append(a)
        self.a_all.append(a)

        # Save to spTrain-trajectory
        self.writer_spTrain.write(a, energy=E, forces=F)

        self.traj_counter += 1
        return E, F

    
    
