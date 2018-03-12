import numpy as np
import scipy as sp
from scipy.spatial.distance import euclidean


def doubleLJ(x, *params):
    """
    Calculates total energy and gradient of N atoms interacting with a
    double Lennard-Johnes potential.
    
    Input:
    x: positions of atoms in form x= [x1,y1,x2,y2,...]
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
    return E, dE


def doubleLJ_energy(x, eps=1.8, r0=1.1, sigma=np.sqrt(0.02), dim=2):
    x = x.reshape((-1, dim))
    E = 0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if j > i:
                r = euclidean(xi, xj)
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
    return E


def doubleLJ_gradient(x, eps=1.8, r0=1.1, sigma=np.sqrt(0.02), dim=2):
    x = x.reshape((-1, dim))
    Natoms = len(x)
    dE = np.zeros(Natoms*dim)
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            r = euclidean(xi,xj)
            if j != i:
                rijVec = xi-xj

                dE1 = 12*rijVec*(-1 / r**14 + 1 / r**8)
                dE2 = eps*(r-r0)*rijVec / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[dim*i:dim*(i+1)] += dE1 + dE2
    return dE

def doubleLJ_energy_ase(a, eps=1.8, r0=1.1, sigma=np.sqrt(0.02)):
    x = a.get_positions()
    E = 0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if j > i:
                r = euclidean(xi, xj)
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
    return E


def doubleLJ_gradient_ase(a, eps=1.8, r0=1.1, sigma=np.sqrt(0.02)):
    x = a.get_positions()
    Natoms, dim = x.shape
    dE = np.zeros((Natoms, dim))
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            r = euclidean(xi,xj)
            if j != i:
                rijVec = xi-xj

                dE1 = 12*rijVec*(-1 / r**14 + 1 / r**8)
                dE2 = eps*(r-r0)*rijVec / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[i] += dE1 + dE2
    return dE


if __name__ == '__main__':
    from ase import Atoms
    
    dim = 2
    x1 = np.array([0,0,
                  0,1,
                  1,0,
                  1,1]) + 0.1*np.random.rand(8)

    
    # Parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    
    print(doubleLJ_energy(x1, eps, r0, sigma))
    print(doubleLJ_gradient(x1, eps, r0, sigma))

    x2 = x1.reshape((-1, dim))
    x2 = np.c_[x2, np.zeros(len(x2))].reshape(-1)

    print(doubleLJ_energy(x2, dim=3))
    print(doubleLJ_gradient(x2, dim=3))

    dim = 3
    L = 2
    d = 1
    positions = x2.reshape((-1,dim))
    pbc = [0,0,0]
    atomtypes = ['H', 'H', 'H', 'H']
    cell = [L,L,d]
    a = Atoms(atomtypes,
              positions=positions,
              cell=cell,
              pbc=pbc)
    
    print(doubleLJ_energy_ase(a))
    print(doubleLJ_gradient_ase(a))
