import os
import sys
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

import time

import numpy as np
cimport numpy as np

from libc.math cimport *  #sqrt, M_PI

from cymem.cymem cimport Pool
cimport cython

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

# Custom functions
ctypedef struct Point:
    double x
    double y
    double z

cdef Point subtract(Point p1, Point p2):
    cdef Point p
    p.x = p1.x - p2.x
    p.y = p1.y - p2.y
    p.z = p1.z - p2.z
    return p

cdef double norm(Point p):
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z)

cdef double euclidean(Point p1, Point p2):
    return norm(subtract(p1,p2))


cdef class Angular_Fingerprint:
    """ comparator for construction of angular fingerprints
    """

    cdef Pool mem
    cdef double Rc1
    cdef double Rc2
    cdef double binwidth1
    cdef double binwidth2
    cdef int Nbins1
    cdef int Nbins2
    cdef double sigma1
    cdef double sigma2
    cdef int nsigma

    cdef double eta
    cdef double gamma
    cdef use_angular

    cdef double volume
    cdef  int Natoms
    cdef int dim

    cdef double m1
    cdef double m2
    cdef double smearing_norm1
    cdef double smearing_norm2

    cdef int Nelements
    def __init__(self, atoms, Rc1=4.0, Rc2=4.0, binwidth1=0.1, Nbins2=30, sigma1=0.2, sigma2=0.10, nsigma=4, eta=1, gamma=3, use_angular=True):
        """ Set a common cut of radius
        """
        self.mem = Pool()

        self.Rc1 = Rc1
        self.Rc2 = Rc2
        self.binwidth1 = binwidth1
        self.Nbins2 = Nbins2
        self.binwidth2 = np.pi / Nbins2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.nsigma = nsigma
        self.eta = eta
        self.gamma = gamma
        self.use_angular = use_angular

        self.volume = atoms.get_volume()
        self.Natoms = atoms.get_number_of_atoms()
        self.dim = 3

        # parameters for the binning:
        self.m1 = self.nsigma*self.sigma1/self.binwidth1  # number of neighbour bins included.
        self.smearing_norm1 = erf(1/np.sqrt(2) * self.m1 * self.binwidth1/self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = self.nsigma*self.sigma2/self.binwidth2  # number of neighbour bins included.
        self.smearing_norm2 = erf(1/np.sqrt(2) * self.m2 * self.binwidth2/self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2

        Nelements_2body = self.Nbins1
        Nelements_3body = self.Nbins2

        if use_angular:
            self.Nelements = Nelements_2body + Nelements_3body
        else:
            self.Nelements = Nelements_2body

    def get_feature(self, atoms):
        """
        """
        cell = atoms.get_cell()
        cdef int Natoms = self.Natoms

        # Get positions and convert to Point-struct
        cdef list pos_np = atoms.get_positions().tolist()
        cdef Point *pos
        pos = <Point*>self.mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            pos[m].x = pos_np[m][0]
            pos[m].y = pos_np[m][1]
            pos[m].z = pos_np[m][2]

        cdef double Rij

        # Initialize feature
        cdef double *feature
        feature = <double*>self.mem.alloc(self.Nelements, sizeof(double))

        cdef int num_pairs, center_bin, minbin_lim, maxbin_lim, newbin
        cdef double normalization, binpos, c, erfarg_low, erfarg_up, value
        cdef int i, j, k
        for i in range(Natoms):
            for j in range(Natoms):
                Rij = euclidean(pos[i], pos[j])

                # Stop if distance too long or atoms are the same one.
                if Rij > self.Rc1+self.nsigma*self.sigma1 or Rij < 1e-6:
                    continue

                # Calculate normalization
                num_pairs = Natoms*Natoms
                normalization = 1./(4*M_PI*Rij*Rij * self.binwidth1 * num_pairs/self.volume * self.smearing_norm1)

                # Identify what bin 'Rij' belongs to + it's position in this bin
                center_bin = <int> floor(Rij/self.binwidth1)
                binpos = Rij/self.binwidth1 - center_bin

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = <int> -ceil(self.m1 - binpos)
                maxbin_lim = <int> ceil(self.m1 - (1-binpos))

                for k in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + k
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 1./sqrt(2)*self.binwidth1/self.sigma1
                    erfarg_low = max(-self.m1, k-binpos)
                    erfarg_up = min(self.m1, k+(1-binpos))
                    value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)

                    # Apply normalization
                    value *= normalization

                    feature[newbin] += value

        # Convert feature to numpy array
        feature_np = np.zeros(self.Nelements)
        for m in range(self.Nelements):
            feature_np[m] = feature[m]

        return feature_np

    def __angle(self, vec1, vec2):
        """
        Returns angle with convention [0,pi]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        arg = np.dot(vec1,vec2)/(norm1*norm2)
        # This is added to correct for numerical errors
        if arg < -1:
            arg = -1.
        elif arg > 1:
            arg = 1.
        return np.arccos(arg), arg

    def __f_cutoff(self, r, gamma, Rc):
        """
        Polinomial cutoff function in the, with the steepness determined by "gamma"
        gamma = 2 resembels the cosine cutoff function.
        For large gamma, the function goes towards a step function at Rc.
        """
        if not gamma == 0:
            return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma
        else:
            return 1

    def __f_cutoff_grad(self, r, gamma, Rc):
        if not gamma == 0:
            return gamma*(gamma+1)/Rc * ((r/Rc)**gamma - (r/Rc)**(gamma-1))
        else:
            return 0

    def angle2_grad(self, RijVec, RikVec):
        Rij = np.linalg.norm(RijVec)
        Rik = np.linalg.norm(RikVec)

        a = RijVec/Rij - RikVec/Rik
        b = RijVec/Rij + RikVec/Rik
        A = np.linalg.norm(a)
        B = np.linalg.norm(b)
        D = A/B

        RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
        RikMat = np.dot(RikVec[:,np.newaxis], RikVec[:,np.newaxis].T)

        a_grad_j = -1/Rij**3 * RijMat + 1/Rij * np.identity(3)
        b_grad_j = a_grad_j

        a_sum_j = np.sum(a*a_grad_j, axis=1)
        b_sum_j = np.sum(b*b_grad_j, axis=1)

        grad_j = 2/(1+D**2) * (1/(A*B) * a_sum_j - A/(B**3) * b_sum_j)



        a_grad_k = 1/Rik**3 * RikMat - 1/Rik * np.identity(3)
        b_grad_k = -a_grad_k

        a_sum_k = np.sum(a*a_grad_k, axis=1)
        b_sum_k = np.sum(b*b_grad_k, axis=1)

        grad_k = 2/(1+D**2) * (1/(A*B) * a_sum_k - A/(B**3) * b_sum_k)


        a_grad_i = -(a_grad_j + a_grad_k)
        b_grad_i = -(b_grad_j + b_grad_k)

        a_sum_i = np.sum(a*a_grad_i, axis=1)
        b_sum_i = np.sum(b*b_grad_i, axis=1)

        grad_i = 2/(1+D**2) * (1/(A*B) * a_sum_i - A/(B**3) * b_sum_i)

        return grad_i, grad_j, grad_k
