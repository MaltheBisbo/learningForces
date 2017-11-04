import numpy as np
from globalOptim_new import globalOptim
from doubleLJ import doubleLJ_energy, doubleLJ_gradient
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
from krr_class_new import krr_class
import sys
import matplotlib.pyplot as plt

def main(arg=1):
    Natoms = 12

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

    Niter = 50
    featureCalculator = fingerprintFeature()
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                        Niter=Niter, Nstag=Niter, sigma=1, saveStep=4)
    optim.runOptimizer()
    Ebest = optim.Ebest
    Xbest = optim.Xbest
    print(optim.Erelaxed)
    # print('Ebest:', Ebest)
    index_groundstate = np.arange(Niter)[optim.Erelaxed < -113.2]
    if len(index_groundstate) == 0:
        Niter_done = np.nan
        Nfev_done = np.nan
        E_done = np.nan
    else:
        index_done = index_groundstate[0]
        Niter_done = int(index_done + 1)
        Nfev_done = optim.Nfev_array[index_done]
        E_done = optim.Erelaxed[index_done]
    
    print('# accepted ML relaxations:', optim.NacceptedML)
    print('Niter_done:', Niter_done)
    print('Nfev_done:', Nfev_done)
    print('Total function evaluations:', optim.Nfev_array[-1])
    print('Training data saved:', optim.ksaved)
    
    np.savetxt('performance_MLenhanced' + str(arg) + '.txt', np.c_[Niter_done, Nfev_done, E_done], delimiter='\t')

    plt.figure(1)
    plt.title('Groundstate for 19 atoms')
    plt.scatter(Xbest[0::2], Xbest[1::2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
if __name__ == '__main__':
    arg = int(sys.argv[1])
    main(arg)
