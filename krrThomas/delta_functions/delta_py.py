import numpy as np
from scipy.spatial.distance import euclidean

class delta():

    def __init__(self, cov_dist=1.0):
        self.cov_dist = cov_dist

    def energy(self, a):
        rmin = 0.7 * self.cov_dist
        radd = 1 - rmin
        x = a.get_positions()
        E = 0
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                if j > i:
                    r = euclidean(xi, xj) + radd
                    E += 1/r**12
        return E

    def forces(self, a):
        rmin = 0.7 * self.cov_dist
        radd = 1 - rmin
        x = a.get_positions()
        Natoms, dim = x.shape
        dE = np.zeros((Natoms, dim))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                if j > i:
                    r = euclidean(xi,xj)
                    r_scaled = r + radd
                    rijVec = xi-xj

                    dE[i] += 12*rijVec*(-1 / (r_scaled**13*r))
                    dE[j] += -12*rijVec*(-1 / (r_scaled**13*r))


        return - dE.reshape(-1)
