import numpy as np
import matplotlib.pyplot as plt
from krr_class import krr_class
from doubleLJ import doubleLJ
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from scipy.optimize import minimize, fmin_bfgs

def makePerturbedGridStructure(Natoms):
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                  if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    x += (np.random.rand(2*Natoms) - 0.5) * 0.6
    return x


def makeStructuresFromRelaxation(Ndata, Natoms, params):
    def LJenergy(x, *params):
        return doubleLJ(x, params[0], params[1], params[2])[0]
    def LJforce(x, *params):
        return -doubleLJ(x, params[0], params[1], params[2])[1]
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x0 = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                   if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    X = np.zeros((Ndata, 2*Natoms))
    k = 0
    while True:
        x = x0 = (np.random.rand(2*Natoms) - 0.5) * 0.6
        res = fmin_bfgs(LJenergy, x, LJforce, args=params,
                        gtol=1e-2,
                        retall=True,
                        full_output=True)
        x_evol = res[7]
        x_add = np.array(x_evol[0::3])
        Nadd = min(Ndata-k, x_add.shape[0])
        print('k:', k)
        X[k:k+Nadd, :] = x_add[:Nadd]
        if k+Nadd < Ndata:
            k += Nadd
        else:
            break
    return X

def makeRandomStructure(Natoms, params):
    eps, r0, sigma = params
    x = np.zeros(2*Natoms)
    while True:
        x = np.random.rand(2*Natoms) * 3
        if doubleLJ(x, eps, r0, sigma)[0] < 0:
            break
    return x


def relaxWithModel(x, model):
    def getEandF(pos):
        E = model.predict_energy(pos=pos)
        F = -model.predict_force()
        return E, F
    res = minimize(getEandF, x,
                   method="TNC",
                   jac=True,
                   tol=1e-1)
    return res.x, res.fun

    
def forceCurve(x, krr_class, coord):
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    unitvec = np.zeros(x.shape[0])
    unitvec[coord] = 1
    Npoints = 1000
    pertub_array = np.linspace(-5, 5, Npoints)
    Xtest = np.array([x + unitvec*pertubation for pertubation in pertub_array])
    curve = np.array([[krr_class.predict_energy(pos=Xtest[i]),
                       doubleLJ(Xtest[i], eps, r0, sigma)[0],
                       pertub_array[i],
                       krr_class.predict_force(pos=Xtest[i])[coord],
                       doubleLJ(Xtest[i], eps, r0, sigma)[1][coord]]
                      for i in range(Npoints) if doubleLJ(Xtest[i], eps, r0, sigma)[0] < 0])

    """
    Epred = np.array([krr_class.predict_energy(pos=xi) for xi in Xtest])
    Etest = np.array([doubleLJ(xi, eps, r0, sigma)[0] for xi in Xtest])
    plt.plot(pertub_array, Etest, color='r')
    plt.plot(pertub_array, Epred, color='b')
    """
    
    #pertub_array[pertub_array > 0] = 'nan'
    plt.scatter(curve[:, 2], curve[:, 1], color='r', s=1)
    plt.scatter(curve[:, 2], curve[:, 0], color='b', s=1)
    plt.scatter(curve[:, 2], curve[:, 3], color='y', s=1)
    plt.scatter(curve[:, 2], curve[:, 4], color='g', s=1)

    plt.show()

def relaxTest(Xtest, model, params):
    params = (1.8, 1.1, np.sqrt(0.02))
    Ntest, Ncoord = Xtest.shape
    Xrelax = np.zeros((Ntest, Ncoord))
    Erelax = np.zeros(Ntest)
    XrelaxLJ = np.zeros((Ntest, Ncoord))
    ErelaxLJ = np.zeros(Ntest)
    for i in range(Ntest):
        print('Progress: {}/{}'.format(i, Ntest))
        Xrelax[i,:], Erelax[i] = relaxWithModel(Xtest[i], model)
        res = minimize(doubleLJ, Xtest[i], params,
                       method="TNC",
                       jac=True,
                       tol=1e-4)
        XrelaxLJ[i,:] = res.x
        ErelaxLJ[i] = res.fun

    Erelax_struct = np.array([doubleLJ(x, params[0], params[1], params[2])[0] for x in Xrelax])
    print('[Erelax, ErelaxLJ]:\n', np.c_[Erelax, ErelaxLJ, Erelax_struct])
    MAE = np.mean(np.abs(ErelaxLJ - Erelax))
    print('MAE relax:', MAE)

def main():
    np.random.seed(555)
    Ndata = 500
    Natoms = 6

    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    lamb = 1e-5
    sig = 10
    
    # X = np.array([makePerturbedGridStructure(Natoms) for i in range(Ndata)])
    X = makeStructuresFromRelaxation(Ndata, Natoms, params)
    featureCalculator = bob_features()
    G, I = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], F[i] = doubleLJ(X[i], eps, r0, sigma)

    Gtrain = G[:-20]
    Etrain = E[:-20]

    # Train model
    comparator = eksponentialComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    GSkwargs = {'lamb': [lamb], 'sigma': [sig]}
    # GSkwargs = {'lamb': np.logspace(-6, -3, 5), 'sigma': np.logspace(-1, 1, 5)}
    print(Etrain.shape, Gtrain.shape)
    MAE, params = krr.gridSearch(Etrain, Gtrain, **GSkwargs)
    print('sigma', params['sigma'])
    print('lamb', params['lamb'])

    print('MAE=', MAE)
    print('MAE using mean:', np.mean(np.fabs(E-np.mean(E))))
    print('Mean absolute energy:', np.mean(np.fabs(E)))

    # forceCurve(X[-1], krr, 6)

    # print('E_unrelaxed\n', E[-20:])
    # Xtest = X[-20:]
    # relaxTest(Xtest, krr, params)

    def createStructuresFromRelaxation(Ndata, *params):
        def LJenergy(x, *params):
            return doubleLJ(x, params[0], params[1], params[2])[0]
        def LJforce(x, *params):
            return -doubleLJ(x, params[0], params[1], params[2])[1]
        X = np.zeros(Ndata)
        Xnotfull = True 
        while Xnotfull:
            
            res = fmin_bfgs(LJenergy, X[-1,:], LJforce, args=params,
                            gtol=1e-2,
                            retall=True,
                            full_output=True)
        
    
    #print('n iterations:', res.nit)
    #print('Eafter', res.fun)
    
    """
    pos = np.reshape(pos, (Natoms, 2))
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()
    """
    
if __name__ == "__main__":
    main()
