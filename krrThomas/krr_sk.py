import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def doubleLJ(x, *params):
    """
    Calculates total energy and gradient of N atoms interacting with a
    double Lennard-Johnes potential.
    
    Input:
    x: positions of atoms in form x= [x1,y2,x2,y2,...]
    params: parameters for the Lennard-Johnes potential
    Output:
    E: Total energy
    dE: gradient of total energy
    """
    eps, r0, sigma = params
    N = x.shape[0]
    Natoms = int(N/2)
    x = np.reshape(x, (Natoms, 2))

    E = 0
    dE = np.zeros(N)
    for i in range(Natoms):
        for j in range(Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            if j > i:
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
            if j != i:
                dxij = x[i, 0] - x[j, 0]
                dyij = x[i, 1] - x[j, 1]

                dEx1 = 12*dxij*(-1 / r**14 + 1 / r**8)
                dEx2 = eps*(r-r0)*dxij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dEy1 = 12*dyij*(-1 / r**14 + 1 / r**8)
                dEy2 = eps*(r-r0)*dyij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[2*i] += dEx1 + dEx2
                dE[2*i + 1] += dEy1 + dEy2
    return E, -dE


class bob_features():
    def __init__(self, X=None):
        """
        --input--
        X:
        Contains data. Each row tepresents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
        """
        self.X = X
        self.calc_featureMat()
    
    def calc_featureMat(self):
        Ndata = self.X.shape[0]
        Natoms = int(self.X.shape[1]/2)
        Nfeatures = int(Natoms*(Natoms-1)/2)
        G = np.zeros((Ndata, Nfeatures))
        I = []
        for n in range(Ndata):
            g, atomIndices = self.calc_singleFeature(self.X[n])
            G[n, :] = g
            I.append(atomIndices)
        self.G = G
        self.I = I
    
    def calc_singleFeature(self, x):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """
        Natoms = int(np.size(x, 0)/2)
        x = x.reshape((Natoms, 2))

        # Calculate inverse bond lengths
        g = np.array([1/np.linalg.norm(x[j]-x[i]) for i in range(Natoms) for j in range(i+1, Natoms)])
        
        # Make list of atom indices corresponding to the elements of the feature g
        atomIndices = [(i, j) for i in range(Natoms) for j in range(i+1, Natoms)]
        
        # Get indices that sort g in decending order
        sorting_indices = np.argsort(-g)
        
        # Sort features and atomic indices
        g_ordered = g[sorting_indices]
        atomIndices_ordered = [atomIndices[i] for i in sorting_indices]
        return g_ordered, atomIndices_ordered


def bobDeriv(pos, g, atomIndices):
    Nr = pos.shape[0]
    pos = pos.reshape((int(Nr/2), 2))
    Nfeatures = np.size(g, 0)
    
    atomIndices = np.array(atomIndices)
    # Calculate gradient of bob-feature
    gDeriv = np.zeros((Nfeatures, Nr))
    for i in range(Nfeatures):
        a0 = atomIndices[i, 0]
        a1 = atomIndices[i, 1]
        inv_r = g[i]
        gDeriv[i, 2*a0:2*a0+2] += inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
        gDeriv[i, 2*a1:2*a1+2] += -inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
    return gDeriv

"""
def gaussKernel(g1, g2, **kwargs):
    sigma = kwargs['sigma']
    d = np.linalg.norm(g2 - g1)
    return np.exp(-1/(2*sigma**2)*d)
"""


def gaussKernel(g1, g2, sigma):
    d = np.linalg.norm(g2 - g1)
    return np.exp(-1/(2*sigma**2)*d) # **2)


class GaussKernel(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0):
        super(GaussKernel, self).__init__()
        self.sigma = sigma

    def transform(self, X):
        return gaussKernel(X, self.X_train_, sigma=self.sigma)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self

def kernelVecDeriv(pos, gnew, inew, G, sig):
    Ntrain, Nfeatures = G.shape
    dvec = np.array([np.linalg.norm(g - gnew) for g in G])
    kappa = np.array([gaussKernel(gnew, g, sig) for g in G])

    dd_dg = np.zeros((Ntrain, Nfeatures))
    for i in range(Ntrain):
        dd_dg[i, :] = -np.sign(G[i] - gnew)
    dg_dR = bobDeriv(pos, gnew, inew)
    dd_dR = np.dot(dd_dg, dg_dR)
    front = -1/(2*sig**2)*kappa  # -1/sig**2*np.multiply(dvec, kappa)
    kernelDeriv = np.multiply(front.reshape((Ntrain, 1)), dd_dR)
    return -kernelDeriv.T

    
def createData(Ndata):
    # Define fixed points
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])
    x = np.c_[x1, x2].reshape((1, 8))
    
    # Define an array of positions for the last pointB
    #xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    xnew = np.c_[np.linspace(0.5, 2, Ndata), np.ones(Ndata)]
    
    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(x, Ndata, axis=0), xnew]
    return X


if __name__ == "__main__":

    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 100
    lamb = 0.0001
    sig = 0.1

    X = createData(Ndata)
    features = bob_features(X)
    G = features.G

    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*5))
    for i in range(Ndata):
        E[i], F[i, :] = doubleLJ(X[i], eps, r0, sigma)

    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)

    
    kernel_params = {'sigma': sig}
    parameters = {'alpha': [lamb, 10*lamb]}
    krr = KernelRidge(alpha=lamb, kernel=gaussKernel, kernel_params=kernel_params)
    estimator = GridSearchCV(krr, param_grid=parameters)
    krr.fit(Gtrain, Etrain-beta)
    

    """
    # Create a pipeline where our custom predefined kernel Chi2Kernel
    # is run before SVC.
    pipe = Pipeline([
        ('chi2', GaussKernel()),
        ('krr', KernelRidge()),
    ])
    print("hello")

    # Set the parameter 'gamma' of our custom kernel by
    # using the 'estimator__param' syntax.
    cv_params = dict([
        ('chi2__sigma', np.array([1, 0.1, 0.01])),
        ('krr__kernel', ['precomputed']),
        ('krr__alpha', np.array([0.001, 0.0001, 0.00001])),
    ])

    # Do grid search to get the best parameter value of 'gamma'.
    model = GridSearchCV(pipe, cv_params, cv=5)
    print('Hellor')
    print(Gtrain.shape)
    print(Etrain.shape)
    model.fit(Gtrain, Etrain)
    Epred = model.predict(G[-1])
    acc_test = accuracy_score(E[-1], Epred)

    print("Test accuracy: {}".format(acc_test))
    print("Best params:")
    print(model.best_params_)
    """
    
    Npoints = 60
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Xtest0 = X[-1]
    delta_array = np.linspace(-3.5, 0.5, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest = Xtest0.copy()
        Xtest[-2] += delta
        Gtest = features.calc_singleFeature(Xtest)[0].reshape((1, -1))
        Etest[i], Ftest = doubleLJ(Xtest, eps, r0, sigma)
        Epredict[i] = krr.predict(Gtest) + beta
    
    plt.plot(delta_array, Etest)
    plt.scatter(delta_array, Epredict)
    #plt.show()

    Xtest = Xtest0.copy()
    Xtest[-2] -= 1.2
    print(Xtest)
    gtest, itest = features.calc_singleFeature(Xtest)
    print(gtest)
    print(itest)
    kappaDeriv = kernelVecDeriv(Xtest, gtest, itest, Gtrain, sig)

    a = krr.dual_coef_

    Fpred = kappaDeriv.dot(a)
    E, Ftest = doubleLJ(Xtest, eps, r0, sigma)
    print(Fpred)
    print(Ftest)
    plt.show()
