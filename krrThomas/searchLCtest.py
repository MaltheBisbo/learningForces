import numpy as np
import matplotlib.pyplot as plt
from globalOptim import globalOptim
from scipy.optimize import minimize
from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
from fingerprintFeature import fingerprintFeature
from eksponentialComparator import eksponentialComparator
from gaussComparator import gaussComparator
from krr_class2 import krr_class
import time

def main():
    X = np.loadtxt('test_searchLC/Xsearch.txt', delimiter='\t')
    E_F = np.loadtxt('test_searchLC/EandFsearch.txt', delimiter='\t')
    E = E_F[:,0]
    F = E_F[:,1:]
    Ndata = len(E)
    Natoms = 7
    
    reg = 1e-6
    sig = 30
    featureCalculator = fingerprintFeature()
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    Nnext = 10

    Ntrain_min = 10
    Ntrain_max = 1000
    Ntrain_points = Ntrain_max-Ntrain_min
    Fpred = np.zeros((Ntrain_points, 2*Natoms*Nnext))
    Epred = np.zeros((Ntrain_points, Nnext))
    MAE = np.zeros(Ntrain_points)
    for i in range(Ntrain_points):
        Ntrain = i+Ntrain_min
        print('i: {}/{}'.format(Ntrain+1, Ntrain_max))
        t0 = time.time()
        GSkwargs = {'reg': [1e-6], 'sigma': [10]}
        MAE[i], params = krr.gridSearch(E[:Ntrain], positionMat=X[:Ntrain], disp=True, **GSkwargs)
        dt = time.time() - t0
        print('dt:', dt)
        
        Epred[i, :] = np.array([krr.predict_energy(pos=x) for x in X[Ntrain:Ntrain+Nnext]])
        Fpred[i, :] = np.array([krr.predict_force(pos=x) for x in X[Ntrain:Ntrain+Nnext]]).reshape(2*Natoms*Nnext)
        

    np.savetxt('test_searchLC/MAEsearchLC.txt', MAE, delimiter='\t')
    np.savetxt('test_searchLC/EpredSearchLC.txt', Epred, delimiter='\t')
    np.savetxt('test_searchLC/FpredSearchLC.txt', Fpred, delimiter='\t')
    """
    plt.figure(1)
    plt.plot(np.arange(Ndata), E)
    plt.show()
    """
if __name__ == '__main__':
    main()
