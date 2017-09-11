import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time


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


def gaussKernel(q1, q2, sigma):
    d = np.linalg.norm(q2 - q1)
    return np.exp(-1/(2*sigma**2)*d**2)


def HessGaussKernel(q1, q2, sigma):
    N = q1.shape[0]
    Hesskernel = np.zeros((N, N))
    for i 

def kernelMat(Q, sigma):
    Ndata = Q.shape[0]
    K = np.zeros((Ndata, Ndata))
    for i in range(Ndata):
        for j in range(i, Ndata):
            K[i, j] = gaussKernel(Q[i, :], Q[j, :], sigma)
    K = K + np.triu(K, k=1)
    return K


def kernelVec(qnew, Q, sigma):
    Ndata = Q.shape[0]
    kappa = np.zeros(Ndata)
    for i in range(Ndata):
        kappa[i] = gaussKernel(qnew, Q[i, :], sigma)
    return kappa


def createData(Ndata):
    # Define fixed points
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])
    x = np.c_[x1, x2].reshape((1, 8))
    
    # Define an array of positions for the last pointB
    xnew = np.c_[np.linspace(-1, 2, Ndata), np.ones(Ndata)]
    
    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(x, Ndata, axis=0), xnew]
    return X

if __name__ == "__main__":

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

    



    

    '''
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
    '''
