import numpy as np
import matplotlib.pyplot as plt
from globalOptim import globalOptim
from scipy.optimize import minimize
from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from gaussComparator import gaussComparator
from krr_class import krr_class

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
    optim = globalOptim(Efun, gradfun, Natoms=13, dmax=1.5, Niter=1000, Nstag=30, sigma=0.5, maxIterLocal=5)
    optim.runOptimizer()
    print('Esaved:\n', optim.Esaved[:optim.ksaved])
    print('best E:', optim.Ebest)
    plotStructure(optim.Xbest)
    plt.show()

def mainML():
    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    sig = 1
    reg = 1e-5
    comparator = gaussComparator(sigma=sig)
    featureCalculator=bob_features()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)
    optim = globalOptim(Efun, gradfun, krr, Natoms=4, dmax=2.5, Niter=50, Nstag=30, sigma=0.5, maxIterLocal=1)
    optim.runOptimizer()

    print(optim.Esaved.T[:optim.ksaved])
    print('best E:', optim.Ebest)
    plt.figure(1)
    plotStructure(optim.Xbest)
    plt.figure(2)
    dErel = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelTrue))
    dErelTrue = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelMLTrue))
    np.savetxt('dErel_vs_ktrain1.txt', np.c_[np.array(optim.ktrain), dErel], delimiter='\t')
    plt.plot(optim.ktrain, dErel, color='r')
    plt.plot(optim.ktrain, dErelTrue, color='b')
    plt.show()
    
    
if __name__ == '__main__':
    mainML()
