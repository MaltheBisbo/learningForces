import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from sklearn.kernel_ridge import KernelRidge

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


def bobFeatures(positions):
    Natoms = int(np.size(positions, 0)/2)
    positions = positions.reshape((Natoms, 2))
    
    # Calculate inverse bond lengths
    g = np.array([1/np.linalg.norm(positions[j]-positions[i]) for i in range(Natoms) for j in range(i+1, Natoms)])

    # Make list of atom indices corresponding to the elements of the feature g
    atomIndices = [(i, j) for i in range(Natoms) for j in range(i+1, Natoms)]

    # Get indices that sort g in decending order
    sorting_indices = np.argsort(-g)
    
    # Sort features and atomic indices
    g_ordered = g[sorting_indices]
    atomIndices_ordered = [atomIndices[i] for i in sorting_indices]
    return g_ordered, atomIndices_ordered


def data2bobFeatures(X):
    """
    --input--
    X:
    Contains data. Each row tepresents a structure given by cartesian coordinates
    in the form [x1, y1, ... , xN, yN]
    """
    Ndata = X.shape[0]
    Natoms = int(X.shape[1]/2)
    Nfeatures = int(Natoms*(Natoms-1)/2)
    G = np.zeros((Ndata, Nfeatures))
    I = []
    for n in range(Ndata):
        g, atomIndices = bobFeatures(X[n])
        G[n, :] = g
        I.append(atomIndices)
    return G, I


def bobDeriv(pos, g, atomIndices):
    Nr = len(atomIndices)
    pos = pos.reshape((int(Nr/2), 2))
    Nfeatures = np.size(g, 0)
    
    atomIndices = np.array(atomIndices)
    # Calculate gradient of bob-feature
    gDeriv = np.zeros((Nfeatures, Nr))
    for i in range(Nfeatures):
        for j in range(Nr):
            a0 = atomIndices[j, 0]
            a1 = atomIndices[j, 1]
            inv_r = g[i]
            gDeriv[i, 2*a0:2*a0+2] += inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
            gDeriv[i, 2*a1:2*a1+2] += -inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
    return gDeriv


def invDistFeature(X):
    Ndata = X.shape[0]
    Natoms = int(X.shape[1]/2)
    Nfeatures = int(Natoms*(Natoms-1)/2)
    Q = np.zeros((Ndata, Nfeatures))
    for n in range(Ndata):
        x = X[n, :].reshape((Natoms, 2))
        k = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                Q[n, k] = 1 / np.linalg.norm(x[i, :] - x[j, :])
                k += 1
    return Q


def gaussKernel(g1, g2, sigma):
    d = np.linalg.norm(g2 - g1)
    return np.exp(-1/(2*sigma**2)*d)  # **)


def HessGaussKernel(g1, g2, sigma):
    N = g1.shape[0]
    Hesskernel = np.zeros((N, N))


def kernelMat(G, sigma):
    Ndata = G.shape[0]
    K = np.zeros((Ndata, Ndata))
    for i in range(Ndata):
        for j in range(i, Ndata):
            K[i, j] = gaussKernel(G[i, :], G[j, :], sigma)
    K = K + np.triu(K, k=1)
    return K


def kernelVec(gnew, G, sigma):
    Ndata = G.shape[0]
    kappa = np.zeros(Ndata)
    for i in range(Ndata):
        kappa[i] = gaussKernel(gnew, G[i, :], sigma)
    return kappa


def kernelVecDeriv(pos, gnew, inew, G, sig):
    Ntrain, Nfeatures = G.shape
    dvec = np.array([np.linalg.norm(g - gnew) for g in G])
    kappa = np.array([gaussKernel(gnew, g, sig) for g in G])
    #print('kappa=\n', kappa)
    dd_dg = np.zeros((Ntrain, Nfeatures))
    for i in range(Ntrain):
        dd_dg[i, :] = -np.sign(G[i] - gnew)
    dg_dR = bobDeriv(pos, gnew, inew)
    dd_dR = np.dot(dd_dg, dg_dR)
    front = -1/(2*sig**2)*kappa  # -1/sig**2*np.multiply(dvec, kappa)
    kernelDeriv = np.multiply(front.reshape((Ntrain, 1)), dd_dR)
    return kernelDeriv.T

    
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

    Ndata = 30
    lamb = 0.001
    sig = 0.1

    X = createData(Ndata)
    G, I = data2bobFeatures(X)

    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*5))
    for i in range(Ndata):
        E[i], F[i, :] = doubleLJ(X[i], eps, r0, sigma)
    
    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)
    print(beta)
    K = kernelMat(Gtrain, sig)
    alpha = np.linalg.inv(K + lamb*np.identity(Ndata-1)).dot(Etrain-beta)
    print(alpha)
    Npoints = 60
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Xtest0 = X[-1]
    delta_array = np.linspace(-3.5, 0.5, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest = Xtest0.copy()
        Xtest[-2] += delta
        Gtest, I = bobFeatures(Xtest)
        kappa = kernelVec(Gtest, Gtrain, sig)
        print(kappa[0], kappa[10], kappa[20])
        Etest[i], F = doubleLJ(Xtest, eps, r0, sigma)
        Epredict[i] = kappa.dot(alpha) + beta
    
    plt.plot(delta_array, Etest)
    plt.scatter(delta_array, Epredict)
    plt.show()
    """
    Gtest = G[-1]
    Itest = I[-1]
    kappa = kernelVec(G[-1], Gtrain, sig)
    print("kappa=\n", kappa)
    print("alpha=\n", alpha)
    
    Epredict = kappa.dot(alpha)
    Etest = E[-1]
    # print(Etrain)
    print("Etest=\n", Etest)
    print("Epredict=\n", Epredict)

    Xtest = X[-1]
    kappaDeriv = kernelVecDeriv(Xtest, Gtest, Itest, Gtrain, sig)

    Fpredict = kappaDeriv.dot(alpha)
    Ftest = F[-1]
    print(Ftest)
    print(Fpredict)
    
    pos_test = Xtest.reshape((5,2))
    plt.scatter(pos_test[:, 0], pos_test[:, 1])
    plt.show()
    """


    """
    # Set interaction parameters
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 100
    lamb = 0.001
    sig = 1
    
    X = createData(Ndata)
    
    # Calculate features F of structures based of atom coordinates X
    Q = invDistFeature(X)
    
    # Calculate energies for each structure
    E = np.zeros(Ndata)
    for i in range(Ndata):
        E[i], F = doubleLJ(X[i], eps, r0, sigma)

    Qtrain = Q[:-1]
    Etrain = E[:-1]
    K = kernelMat(Qtrain, sig)
        
    alpha = np.linalg.inv(K + lamb*np.identity(Ndata-1)).dot(Etrain)

    kappa = kernelVec(Q[-1], Q[:-1], sig)

    Epredict = kappa.dot(alpha)
    Etest = E[-1]
    print(Etrain)
    print(Etest)
    print(Epredict)
    """
    



    


    r = np.linspace(0.8, 2.5, 100)
    x1 = np.array([0, 0])
    x2 = np.c_[r, np.zeros(100)]
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    E = np.zeros(100)
    Fx = np.zeros(100)
    for i in range(100):
        X = np.array([x1, x2[i, :]])
        print(X)
        E[i], F = doubleLJ(X, eps, r0, sigma)
        Fx[i] = F[0]
    plt.plot(r, E)
    plt.plot(r, Fx)
    plt.xlim([0.9, 2.5])
    plt.ylim([-10, 10])
    plt.show()

