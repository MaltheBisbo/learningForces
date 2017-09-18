import numpy as np
import matplotlib.pyplot as plt
from krr_class import doubleLJ, bob_features, krr_class, gaussComparator
from scipy.optimize import minimize


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

def makeRandomStructure(Natoms):
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.9
    x = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                  if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    x += (np.random.rand(2*Natoms) - 0.5)
    return x
    
def main():

    Ndata = 1000
    Natoms = 5

    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    # parameters for kernel and regression
    lamb = 0.0001
    sig = 0.1
    
    X = np.array([makeRandomStructure(Natoms) for i in range(Ndata)])
    featureCalculator = bob_features()
    G, I = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], F[i] = doubleLJ(X[i], eps, r0, sigma)

    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)
        
    # Train model
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    krr.fit(Etrain, Gtrain, lamb=lamb)

    print(G[-1])
    print(G.shape)
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
