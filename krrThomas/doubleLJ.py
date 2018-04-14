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

class doubleLJ_ase():

    def __init__(self, eps=1.8, r0=1.1, sigma=np.sqrt(0.02)):
        self.eps = eps
        self.r0 = r0
        self.sigma = sigma

    def energy(self, a):
        eps, r0, sigma = self.eps, self.r0, self.sigma
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

    def forces(self, a):
        eps, r0, sigma = self.eps, self.r0, self.sigma
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
        return - dE.reshape(-1)

class delta_ase():

    def __init__(self, eps=1.8, r0=1.1, sigma=np.sqrt(0.02), cov_dist=1.0):
        self.eps = eps
        self.r0 = r0
        self.sigma = sigma
        self.cov_dist = cov_dist

    def energy(self, a):
        rmin = 0.7 * self.cov_dist
        radd = 1 - rmin
        eps, r0, sigma = self.eps, self.r0, self.sigma
        x = a.get_positions()
        E = 0
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                if j > i:
                    r = euclidean(xi, xj) + radd
                    E1 = 1/r**12 #- 2/r**6                                                                              
                    #E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                    E += E1 #+ E2
        return E

    def forces(self, a):
        rmin = 0.7 * self.cov_dist
        radd = 1 - rmin
        eps, r0, sigma = self.eps, self.r0, self.sigma
        x = a.get_positions()
        Natoms, dim = x.shape
        dE = np.zeros((Natoms, dim))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                r = euclidean(xi,xj)
                r_scaled = r + radd
                if j != i:
                    rijVec = xi-xj
                    
                    dE1 = 12*rijVec*(-1 / (r_scaled**13*r))# + 1 / r**8)
                    #dE2 = eps*(r-r0)*rijVec / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))
                 
                    dE[i] += dE1 #+ dE2
        return - dE.reshape(-1)


if __name__ == '__main__':
    from ase import Atoms
    import matplotlib.pyplot as plt
    
    dim = 2
    x1 = np.array([0,0,
                  0,1,
                  1,0,
                  1,1]) + 0.1*np.random.rand(8)

    
    # Parameters for potential
    eps, r0, sigma = 1.0, 1.2, np.sqrt(0.2)
    
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


    # Plot potential

    eps, r0, sigma, cov_dist = 1.0, 1.8, np.sqrt(0.3), 1
    delta = delta_ase(eps, r0, sigma)

    x0 = np.array([0, 0, 0])

    Npoints = 100
    r = np.linspace(0.55, 3.5, Npoints)
    X = np.c_[np.zeros((Npoints,5)), r]

    atoms_list = []
    for x in X:
        a = Atoms(['H','H'],
                  positions=x.reshape((-1,dim)),
                  cell=cell,
                  pbc=pbc)
        atoms_list.append(a)
    
    
    E = np.array([delta.energy(a) for a in atoms_list])
    F = np.array([delta.forces(a)[-1] for a in atoms_list])
    
    dr = r[1] - r[0]
    F_num = - (E[1:] - E[:-1]) / dr
    r_forF = r[:-1] + dr/2

    plt.figure()
    plt.plot(r, E)
    plt.xlabel('r')
    plt.ylabel('E')

    plt.figure()
    plt.plot(r, F)
    plt.plot(r_forF, F_num, 'k:')
    

    
    plt.show()

