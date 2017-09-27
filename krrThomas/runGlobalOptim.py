import numpy as np
import matplotlib.pyplot as plt
from globalOptim import globalOptim
from scipy.optimize import minimize
from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from gaussComparator import gaussComparator

def plotStructure(X, boxsize=None, color='b'):
    Natoms = int(X.shape[0]/2)
    if boxsize is None:
        boxsize = 1.5 * np.sqrt(Natoms)

    xbox = np.array([0, boxsize, boxsize, 0, 0])
    ybox = np.array([0, 0, boxsize, boxsize, 0])
    
    x = X[0::2]
    y = X[1::2]
    plt.plot(xbox, ybox, color='k')
    plt.scatter(x, y, color=color, s=8)
    plt.gca().set_aspect('equal', adjustable='box')

    
def testLocalMinimizer():
    def Efun(X):
        params = (1, 1.4, np.sqrt(0.02))
        return doubleLJ(X, params[0], params[1], params[2])
    optim = globalOptim(Efun, Natoms=20)
    optim.makeInitialStructure()
    Erel, Xrel = optim.relax()
    plotStructure(Xrel, color='r')
    print('E:', Erel)
    plt.show()


def main():
    def Efun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])
    optim = globalOptim(Efun, gradfun, Natoms=13, dmax=1.5, Niter=50, Nstag=10, sigma=0.5, maxfev=10)
    optim.runOptimizer()
    print('Esaved:\n', optim.Esaved[:optim.ksaved])
    print('best E:', optim.Ebest)
    plotStructure(optim.Xbest)
    plt.show()

def mainML():
    def Efun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    sig = 10 
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=bob_features)
    optim = globalOptim(Efun, gradfun, Natoms=26, dmax=2.5, Niter=50, Nstag=10, sigma=0.5, maxfev=100)
    optim.runOptimizer()
    print('best E:', optim.Ebest)
    plotStructure(optim.Xbest)
    plt.show()
    
    
if __name__ == '__main__':
    main()
