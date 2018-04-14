import numpy as np
cimport numpy as np

from libc.math cimport *

from cymem.cymem cimport Pool
cimport cython

# Custom functions
ctypedef struct Point:
    double coord[3]

cdef Point subtract(Point p1, Point p2):
    cdef Point p
    p.coord[0] = p1.coord[0] - p2.coord[0]
    p.coord[1] = p1.coord[1] - p2.coord[1]
    p.coord[2] = p1.coord[2] - p2.coord[2]
    return p

cdef double norm(Point p):
    return sqrt(p.coord[0]*p.coord[0] + p.coord[1]*p.coord[1] + p.coord[2]*p.coord[2])

cdef double euclidean(Point p1, Point p2):
    return norm(subtract(p1,p2))

class delta():

    def __init__(self, cov_dist=1.0):
        self.cov_dist = cov_dist

    def energy(self, a):

        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cdef int Natoms = a.get_number_of_atoms()

        cdef list x_np = a.get_positions().tolist()
        cdef Point *x
        x = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            x[m].coord[0] = x_np[m][0]
            x[m].coord[1] = x_np[m][1]
            x[m].coord[2] = x_np[m][2]

        cdef double rmin = 0.7 * self.cov_dist
        cdef double radd = 1 - rmin

        E = 0
        cdef int i, j
        for i in range(Natoms):
            xi = x[i]
            for j in range(Natoms):
                xj = x[j]
                if j > i:
                    r_scaled = euclidean(xi, xj) + radd
                    E += 1/pow(r_scaled,12)
        return E

    def forces(self, a):
        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cdef int Natoms = a.get_number_of_atoms()
        cdef int dim = 3

        cdef list x_np = a.get_positions().tolist()
        cdef Point *x
        x = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            x[m].coord[0] = x_np[m][0]
            x[m].coord[1] = x_np[m][1]
            x[m].coord[2] = x_np[m][2]

        cdef double rmin = 0.7 * self.cov_dist
        cdef double radd = 1 - rmin

        # Initialize Force object
        cdef double *dE
        dE = <double*>mem.alloc(Natoms * dim, sizeof(double))
        cdef int i, j, k
        for i in range(Natoms):
            xi = x[i]
            for j in range(Natoms):
                xj = x[j]

                r = euclidean(xi,xj)
                r_scaled = r + radd
                if j != i:
                    rijVec = subtract(xi,xj)

                    for k in range(dim):
                        dE[dim*i + k] = -12*rijVec.coord[k] / (r_scaled**13*r)

        dE_np = np.zeros((Natoms, dim))
        for i in range(Natoms):
            for k in range(dim):
                dE_np[i,k] = dE[dim*i + k]

        return - dE_np.reshape(-1)
