import numpy as np
import matplotlib.pyplot as plt
from krr_class_new import krr_class
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
import time

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

    permutation = np.random.permutation(Ndata)
    pos = pos[permutation]
    E = E[permutation]
    F = F[permutation]
    print('Data loaded')
    return pos.reshape((Ntraj, Na*dim)), E, F.reshape((Ntraj, Na*dim))

def energyANDforceLC():
    #np.random.seed(455)
    Ndata = 1500
    Natoms = 24
    
    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    reg = 1e-7
    sig = 30
    X, E, F = loadTraj(Ndata)
    featureCalculator = fingerprintFeature(dim=3)
    t0 = time.time()
    G = featureCalculator.get_featureMat(X)
    print('Time to calculate features:', time.time() - t0)

    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    NpointsLC = 10
    Ndata_array = np.logspace(1,3,NpointsLC).astype(int)
    FVU_energy_array = np.zeros(NpointsLC)
    FVU_force_array = np.zeros((NpointsLC, 2*Natoms))
    for i in range(NpointsLC):
        N = int(3/2*Ndata_array[i])
        Esub = E[:N]
        Fsub = F[:N]
        Xsub = X[:N]
        Gsub = G[:N]
        t0 = time.time()
        GSkwargs = {'reg': [1e-7], 'sigma': np.logspace(0,2,6)}
        #FVU_energy_array[i], FVU_force_array[i, :] = krr.train(Esub, Fsub, Gsub, Xsub, reg=reg)
        FVU_energy_array[i], params = krr.train(Esub, featureMat=Gsub, positionMat=Xsub, add_new_data=False, **GSkwargs)
        print('Ntrain:', N)
        print('dt:', time.time() - t0)
        print('params:', params)
        print('FVU_energy: {}\n'.format(FVU_energy_array[i]))

    #np.savetxt('grapheneLCdata.txt', np.c_[Ndata_array, FVU_energy_array, FVU_force_array], delimiter='\t')
    plt.figure(1)
    plt.title('Energy learning curve (random structures)')
    plt.loglog(Ndata_array, FVU_energy_array)
    plt.xlabel('# training data')
    plt.ylabel('unexplained variance')
    """
    plt.figure(2)
    plt.title('Force learning curve (random structures)')
    plt.loglog(Ndata_array, FVU_force_array)
    plt.xlabel('# training data')
    plt.ylabel('unexplained variance')
    """
    plt.show()

if __name__ == "__main__":
    #energyLC()
    energyANDforceLC()
