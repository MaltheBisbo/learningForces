import numpy as np
from krr_class2 import krr_class
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
from scipy.optimize import minimize, fmin_bfgs
import time
import sys

def makeConstrainedStructure(Natoms):
    boxsize = 1.5 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 1.5
    def validPosition(X, xnew):
        Natoms = int(len(X)/2) # Current number of atoms                                                                            
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[2*i:2*i+2])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    Xinit = np.zeros(2*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.random.rand(2) * boxsize
            if validPosition(Xinit[:2*i], xnew):
                Xinit[2*i:2*i+2] = xnew
                break
    return Xinit

def energyANDforceLC(arg=1):
    Ndata = 1500
    Natoms = 7
    
    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    reg = 1e-7
    sig = 30
    
    X = np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])
    featureCalculator = fingerprintFeature()
    G = featureCalculator.get_featureMat(X)

    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

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
        t0 = time.time()
        GSkwargs = {'reg': [reg], 'sigma': np.logspace(0, 2, 5)}
        FVU_energy_array[i], FVU_force_array[i, :], _ = krr.gridSearch_EandF(Esub, Fsub, Gsub, Xsub, **GSkwargs)
        print('dt:', time.time() - t0)
        print(FVU_energy_array[i])

    np.savetxt('LC_finger_random' + str(arg) + '.dat', np.c_[Ndata_array, FVU_energy_array, FVU_force_array], delimiter='\t')

if __name__ == "__main__":
    arg = int(sys.argv[1])
    energyANDforceLC(arg)
