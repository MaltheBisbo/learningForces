import random
import numpy as np
cimport numpy as np

from libc.math cimport sqrt

from cymem.cymem cimport Pool
cimport cython
"""
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


cdef double* subtract(double* p1, double* p2):
    cdef double* p
    p[0] = p1[0] - p2[0]
    p[1] = p1[1] - p2[1]
    p[2] = p1[2] - p2[2]
    return p

cdef double norm(double* p):
    return sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

cdef double euclidean(double* p1, double* p2):
    return norm(subtract(p1,p2))
"""
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

@cython.boundscheck(False)
cpdef double test(int N):
    cdef Pool mem
    mem = Pool()

    np.random.seed(42)
    pos_np = np.random.rand(N,3).tolist()

    cdef Point *pos
    pos = <Point*>mem.alloc(N, sizeof(Point))
    cdef int k
    for k in range(N):
        pos[k].coord[0] = pos_np[k][0]
        pos[k].coord[1] = pos_np[k][1]
        pos[k].coord[2] = pos_np[k][2]
    cdef double Rij = 0

    cdef int i, j
    for i in range(N):
        for j in range(N):
            Rij += euclidean(pos[i], pos[j])
    return Rij
"""
@cython.boundscheck(False)
cpdef double test(int N):
    cdef np.ndarray[np.double_t, ndim=2] pos
    pos = np.random.rand(N,3)
    #cdef list pos = np.random.rand(N,2).tolist()
    cdef double Rij = 0

    cdef int i, j
    for i in range(N):
        for j in range(N):
            Rij += (pos[i,0] - pos[j,0])*(pos[i,0] - pos[j,0]) + (pos[i,1] - pos[j,1])*(pos[i,1] - pos[j,1]) + (pos[i,2] - pos[j,2])*(pos[i,2] - pos[j,2])
    return Rij
"""
