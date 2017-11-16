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

    X = np.loadtxt('work_folder/graphene_all_positions.txt', delimiter='\t')
    G = np.loadtxt('work_folder/graphene_all_features.txt', delimiter='\t')
    E = np.loadtxt('work_folder/graphene_all_Energies.txt', delimiter='\t')
    F = np.loadtxt('work_folder/graphene_all_Forces.txt', delimiter='\t')

    # Find local extremas in E
    idx_maxE = argrelextrema(E, np.greater, order=3)[0]
    idx_maxE = np.r_[0, idx_maxE]
    #print(idx_maxE)
    #plt.plot(np.arange(len(E)), E)
    #plt.show()

    featureCalculator = fingerprintFeature(dim=3, rcut=6, binwidth=0.1, sigma=0.3)
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    idx = np.arange(len(E))
    idx_traj = np.split(np.arange(len(E)), idx_maxE[1:])
    Ntraj = len(idx_traj)
    idx_minE = np.zeros(Ntraj)
    idx_rest = np.zeros(len(E) - 2*Ntraj)
    k = 0
    for i, traj in enumerate(idx_traj[0]):
        print(traj)
        len_traj = len(traj)
        idx_minE[i] = traj[0]
        
        idx_rest[k:len_traj-2] = traj[1:-1]
        k += len_traj - 2

    print(len(idx_maxE))
    print(len(idx_minE))
    print(len(idx_rest))
    print(len(E))
        
    idx_traj = np.array([list(index) for index in idx_traj])
    print(idx_traj.shape)
    
    Ntrain = 1000
    i_traj = 3
    idx_relax = idx_maxE[i_traj]
    idx_minE = idx_maxE[1:] - 1
    idx_min_train  = np.r_[idx_minE[:i_traj], idx_minE[i_traj+1:]]
    [idx_train1, idx_test, idx_train2] = np.split(np.arange(len(E)), [idx_maxE[i], idx_maxE[i+1]]) 
    idx_train = np.r_[idx_train1, idx_train2]
    idx_train_sub = np.random.permutation(idx_train)[:Ntrain]
        
    Esub = E[idx_train_sub]
    Fsub = F[idx_train_sub]
    Xsub = X[idx_train_sub]
    Gsub = G[idx_train_sub]

    GSkwargs = {'reg': [1e-7], 'sigma': np.logspace(0,2,6)}
    #FVU_energy_array[i], FVU_force_array[i, :] = krr.train(Esub, Fsub, Gsub, Xsub, reg=reg)
    FVU_E, params = krr.train(Esub, featureMat=Gsub, positionMat=Xsub, add_new_data=False, **GSkwargs)
    print('params:', params)
    print('FVU_energy: {}\n'.format(FVU_E))

    label='grapheneMLrelax/grapheneML'
    calculator = krr_calculator(krr, label)

    x_to_relax = X[idx_maxE[3]].reshape((-1,3))
    a = Atoms('C24', x_to_relax)
    a.set_calculator(calculator)
    dyn = BFGS(a, trajectory='grapheneMLrelax/graphene_1000RestRand.traj')
    dyn.run(fmax=0.1)

if __name__ == "__main__":
    main()
