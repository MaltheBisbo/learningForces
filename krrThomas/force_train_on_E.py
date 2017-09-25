import numpy as np
import matplotlib.pyplot as plt
from krr_class import krr_class
from doubleLJ import doubleLJ
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from scipy.optimize import minimize


def makePerturbedGridStructure(Natoms):
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                  if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    x += (np.random.rand(2*Natoms) - 0.5) * 0.6
    return x


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
                       tol=1e-2)
        XrelaxLJ[i,:] = res.x
        ErelaxLJ[i] = res.fun

    Etrue = np.array([doubleLJ(x, params[0], params[1], params[2])[0] for x in Xrelax])
    print('[Erelax, ErelaxLJ]:\n', np.c_[Erelax, ErelaxLJ, Etrue])
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
    
    X = np.array([makePerturbedGridStructure(Natoms) for i in range(Ndata)])
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

    forceCurve(X[-1], krr, 6)

    print('E_unrelaxed\n', E[-20:])
    Xtest = X[-20:]
    relaxTest(Xtest, krr, params)
    
    """
    pos = np.reshape(X, (Natoms, 2))
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()
    """
    
if __name__ == "__main__":
    main()
