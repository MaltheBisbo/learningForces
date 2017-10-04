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

    sig = 3
    reg = 1e-5
    comparator = gaussComparator(sigma=sig)
    featureCalculator = bob_features()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)
    optim = globalOptim(Efun, gradfun, krr, Natoms=7, dmax=2.5, Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
    optim.runOptimizer()

    GSkwargs = {'reg': [reg], 'sigma': [sig]}
    MAE, params = krr.gridSearch(optim.Esaved[:optim.ksaved], positionMat=optim.Xsaved[:optim.ksaved], **GSkwargs)
    print('MAE:', MAE)
    print(optim.Esaved.T[:optim.ksaved])
    print('best E:', optim.Ebest)
    # Make into numpy arrays
    ktrain = np.array(optim.ktrain)
    EunrelML = np.array(optim.EunrelML)
    Eunrel = np.array(optim.Eunrel)
    FunrelML = np.array(optim.FunrelML)
    FunrelTrue = np.array(optim.FunrelTrue)
    
    dErel = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelTrue))
    dErelTrue = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelMLTrue))
    dE2rel = np.fabs(np.array(optim.ErelML) - np.array(optim.E2rel))
    dEunrel = np.fabs(EunrelML - Eunrel)
    
    np.savetxt('dErel_vs_ktrain2.txt', np.c_[ktrain, dErel, dErelTrue, EunrelML, Eunrel], delimiter='\t')

    print(FunrelTrue)
    print(FunrelTrue.shape)
    
    plt.figure(1)
    plt.title('Structure Example')
    plotStructure(optim.Xbest)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure(2)
    plt.title('Energies of relaxed struxtures')
    plt.loglog(ktrain, dErel, color='r')
    plt.loglog(ktrain, dErelTrue, color='b')
    plt.loglog(ktrain, dE2rel, color='g')
    plt.xlabel('NData')
    plt.ylabel('Energy')
    # plt.legend()
    plt.figure(3)
    plt.title('Energies of unrelaxed structures')
    plt.loglog(ktrain, dEunrel)
    plt.xlabel('NData')
    plt.ylabel('Energy')
    plt.figure(4)
    plt.title('Forces of unrelaxed structures (ML)')
    plt.loglog(ktrain, FunrelML)
    plt.xlabel('NData')
    plt.ylabel('Force')
    plt.figure(5)
    plt.title('Forces of unrelaxed structures (True)')
    plt.loglog(ktrain, FunrelTrue)
    plt.xlabel('NData')
    plt.ylabel('Force')
    plt.show()
    
def mainTestLearning():
    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])

    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    sig = 3
    reg = 1e-5
    Natoms = 7
    comparator = gaussComparator(sigma=sig)
    featureCalculator = bob_features()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)
    
    Nruns = 50
    Nstructs = 1200
    optimData = np.zeros((Nruns, Nstructs, 2*Natoms))
    for i in range(Nruns):
        print('Data creation: {}/{}'.format(i, Nruns))
        optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                            Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
        optim.runOptimizer()
        optimData[i,:] = optim.Xsaved[:Nstructs]
    print('optimData created with shape:', optimData.shape)
    optimData = np.array(optimData)
    
    Npoints = 20
    FVU_energy = np.zeros(Npoints)
    FVU_force = np.zeros((Npoints, 2*Natoms))
    Ntrain_array = np.logspace(1, 3, Npoints).astype(int)
    for i in range(Nruns):
        E = np.zeros(Nstructs)
        F = np.zeros((Nstructs, 2*Natoms))
        for s in range(Nstructs):
            E[s], F[s,:] = doubleLJ(optimData[i,s], 1.8, 1.1, np.sqrt(0.02))
        for n in range(Npoints):
            print('i:{}/{} , n:{}/{}'.format(i,Nruns,n,Npoints))
            Ntrain = Ntrain_array[n]
            Ntest = int(max(10, np.round(Ntrain/5)))
            krr.fit(E[:Ntrain], positionMat=optimData[i, :Ntrain, :])
            Etest = E[Ntrain:Ntrain+Ntest]
            Ftest = F[Ntrain:Ntrain+Ntest]
            FVU_energy[n] += krr.get_MAE_energy(Etest, positionMat=optimData[i, Ntrain:Ntrain+Ntest, :])
            FVU_force[n,:] = krr.get_MAE_force(Ftest, positionMat=optimData[i, Ntrain:Ntrain+Ntest, :])
    FVU_energy /= Nruns
    FVU_force /= Nruns

    np.savetxt('LC_search_bob_N7.txt', np.c_[Ntrain_array, FVU_energy, FVU_force], delimiter='\t')
    plt.figure(1)
    plt.title('Energy FVU vs training size (from search)')
    plt.loglog(Ntrain_array, FVU_energy)
    plt.xlabel('# training data')
    plt.ylabel('FVU')
    plt.figure(2)
    plt.title('Force FVU vs training size (from search)')
    plt.loglog(Ntrain_array, FVU_force)
    plt.xlabel('# training data')
    plt.ylabel('FVU')
    plt.show()
            
if __name__ == '__main__':
    #mainML()
    mainTestLearning()
