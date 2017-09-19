import numpy as np
import matplotlib.pyplot as plt
from krr_class import doubleLJ, bob_features, krr_class, gaussComparator
from scipy.optimize import minimize

"""
def randomData(Natoms, params, boxsize=None):
    if boxsize is None:
        boxsize = 1.5*np.sqrt(Natoms)
        
    bounds = [(0, boxsize)] * Natoms * 2

    E = 0
    x = np.zeros(2*Natoms)
    running = True
    while running:
        x0 = np.random.rand(Natoms, 2) * boxsize
        res = minimize(doubleLJ, x0, params,
                       method="TNC",
                       jac=True,
                       tol=1e-0,
                       bounds=bounds)
        if res.fun < 0:
            running = False
            E = res.fun
            x = res.x
    print('E=', E)
    return x
"""

def makeRandomStructure(Natoms):
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                  if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    x += (np.random.rand(2*Natoms) - 0.5) * 0.6
    return x


def forceCurve(x, krr_class):
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    unitvec = np.zeros(x.shape[0])
    unitvec[13] = 1
    Npoints = 200
    pertub_array = np.linspace(-5, 5, Npoints)
    Xtest = np.array([x + unitvec*pertubation for pertubation in pertub_array])
    curve = np.array([[krr_class.predict_energy(pos=Xtest[i]), doubleLJ(Xtest[i], eps, r0, sigma)[0], pertub_array[i],
                       krr_class.predict_force(pos=Xtest[i])[13], doubleLJ(Xtest[i], eps, r0, sigma)[1][13]]
                      for i in range(Npoints) if doubleLJ(Xtest[i], eps, r0, sigma)[0] < 0])
    """
    Epred = np.array([krr_class.predict_energy(pos=xi) for xi in Xtest])
    Etest = np.array([doubleLJ(xi, eps, r0, sigma)[0] for xi in Xtest])
    plt.plot(pertub_array, Etest, color='r')
    plt.plot(pertub_array, Epred, color='b')
    """
    plt.scatter(curve[:, 2], curve[:, 1], color='r', s=2)
    plt.scatter(curve[:, 2], curve[:, 0], color='b', s=2)
    plt.scatter(curve[:, 2], curve[:, 3], color='y', s=2)
    plt.scatter(curve[:, 2], curve[:, 4], color='g', s=2)

    plt.show()


def main():
    np.random.seed(555)
    Ndata = 100
    Natoms = 15

    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    # parameters for kernel and regression
    lamb = 0.001
    sig = 3
    
    X = np.array([makeRandomStructure(Natoms) for i in range(Ndata)])
    featureCalculator = bob_features()
    G, I = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], F[i] = doubleLJ(X[i], eps, r0, sigma)

    Gtrain = G[:-1]
    Etrain = E[:-1]

    # Train model
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    #GSkwargs = {'lamb': [0.001, 0.0001], 'sigma': [3, 1]}
    GSkwargs = {'lamb': np.logspace(-6, -3, 10), 'sigma': np.logspace(-1, 1, 5)}
    print(Etrain.shape, Gtrain.shape)
    MAE, params = krr.gridSearch(Etrain, Gtrain, **GSkwargs)
    print('sigma', params['sigma'])
    print('lamb', params['lamb'])
    #MAE = krr.cross_validation(E, G, lamb=lamb)
    #krr.fit(Etrain, Gtrain, lamb=lamb)

    print('MAE=', MAE)
    print('MAE using mean:', np.mean(np.fabs(E-np.mean(E))))
    print('Mean absolute energy:', np.mean(np.fabs(E)))

    forceCurve(X[-1], krr)
    
    Etest = E[-1]
    Epred = krr.predict_energy(pos=X[-1])
    Ftest = F[-1]
    Fpred = krr.predict_force()
    
    print('Etest=\n', Etest)
    print('Epred=\n', Epred)
    print('Ftest=\n', Ftest)
    print('Fpred=\n', Fpred)
    
    
    """
    pos = np.reshape(X, (Natoms, 2))
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()
    """
    
if __name__ == "__main__":
    main()
