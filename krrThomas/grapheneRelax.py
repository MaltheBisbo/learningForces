import numpy as np
import matplotlib.pyplot as plt
from krr_class_new import krr_class
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from scipy.signal import argrelextrema
from krr_calculator import krr_calculator
import time

from ase import Atoms
from ase.optimize import BFGS
from ase.io import read, write

def loadTraj(Ndata):
    atoms = read('work_folder/all.traj', index=':')
    atoms = atoms[0:15000]
    atoms = atoms[::10]
    atoms = atoms[:Ndata]
    Na = 24
    dim = 3
    Ntraj = len(atoms)
    
    pos = np.zeros((Ntraj,Na,dim))
    E = np.zeros(Ntraj)
    F = np.zeros((Ntraj, Na, dim))
    for i, a in enumerate(atoms):
        pos[i] = a.positions
        E[i] = a.get_potential_energy()
        F[i] = a.get_forces()

    permutation = np.random.permutation(Ntraj)
    pos = pos[permutation]
    E = E[permutation]
    F = F[permutation]
    print('Data loaded')
    return pos.reshape((Ntraj, Na*dim)), E, F.reshape((Ntraj, Na*dim))

def main():
    #np.random.seed(455)
    Ndata = 1500
    Natoms = 24
    
    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    reg = 1e-7
    sig = 30

    Nstructs = len(E)
    
    # Find local extremas in E
    idx_maxE = argrelextrema(E, np.greater, order=3)[0]

    idx_traj = np.split(np.arange(len(E)), idx_maxE)
    print(idx_traj[0].shape)
    Ntraj = len(idx_traj)
    print(Ntraj)
    idx_minE = np.zeros(Ntraj).astype(int)
    idx_rest = np.zeros(Nstructs - 2*Ntraj).astype(int)
    k=0
    for i in range(Ntraj):
        traj = idx_traj[i]
        len_traj = len(traj)
        idx_minE[i] = traj[-1]
        idx_rest[k:k+len_traj-2] = traj[1:-1]
        k += len_traj - 2
    idx_maxE = np.r_[0, idx_maxE]

    Ntrain = 1000
    idx_rest_permut = np.random.permutation(idx_rest)
    idx_train = np.r_[idx_minE, idx_rest_permut[:Ntrain - Ntraj]]
    print(idx_train.shape)
    
    featureCalculator = fingerprintFeature(dim=3, rcut=6, binwidth=0.1, sigma=0.3)
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    Esub = E[idx_train]
    Fsub = F[idx_train]
    Xsub = X[idx_train]
    Gsub = G[idx_train]

    GSkwargs = {'reg': np.logspace(-1, -7, 10), 'sigma': np.logspace(-1,2,10)}
    print(GSkwargs)
    #FVU_energy_array[i], FVU_force_array[i, :] = krr.train(Esub, Fsub, Gsub, Xsub, reg=reg)
    FVU_E, params = krr.train(Esub, featureMat=Gsub, positionMat=Xsub, add_new_data=False, **GSkwargs)
    print('params:', params)
    print('FVU_energy: {}\n'.format(FVU_E))

    label='grapheneMLrelax/grapheneML'
    calculator = krr_calculator(krr, label)
    
    x_to_relax = X[idx_maxE[10]].reshape((-1,3))
    a = Atoms('C24', x_to_relax)
    a.set_calculator(calculator)
    dyn = BFGS(a, trajectory='grapheneMLrelax/graphene10.traj')
    dyn.run(fmax=0.1)

if __name__ == "__main__":
    main()
