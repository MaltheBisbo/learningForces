import numpy as np
from globalOptim_gr2 import globalOptim
from scipy.optimize import minimize
from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
from krr_class2 import krr_class
import sys

def energyANDforceLC_searchData(arg=1):
    Ndata = 1500
    Natoms = 7

    # parameters for potential                                                                                                          
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    # parameters for kernel and regression                                                                                              
    reg = 1e-7
    sig = 30

    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])

    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    featureCalculator = fingerprintFeature()
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                        Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
    optim.runOptimizer()
    X = optim.Xsaved[:Ndata]
    X = np.random.permutation(X)
    
    G = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i] = -grad
    
    NpointsLC = 30
    Ndata_array = np.logspace(1,3,NpointsLC).astype(int)
    FVU_energy_array = np.zeros(NpointsLC)
    FVU_force_array = np.zeros((NpointsLC, 2*Natoms))
    for i in range(NpointsLC):
        N = int(3/2*Ndata_array[i])
        Esub = E[:N]
        Fsub = F[:N]
        Xsub = X[:N]
        Gsub = G[:N]
        GSkwargs = {'reg': [reg], 'sigma': np.logspace(0, 2, 5)}
        FVU_energy_array[i], FVU_force_array[i, :], _ = krr.gridSearch_EandF(Esub, Fsub, Gsub, Xsub, **GSkwargs)
        print(FVU_energy_array[i])
        #print(FVU_force_array[i])

    np.savetxt('LC_finger_search_perm' + str(arg) + '.dat', np.c_[Ndata_array, FVU_energy_array, FVU_force_array], delimiter='\t')

if __name__ == '__main__':
    arg = int(sys.argv[1])
    energyANDforceLC_searchData(arg)
