import random
import numpy as np
cimport numpy as np

from libc.math cimport sqrt

from cymem.cymem cimport Pool
cimport cython




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

def convert_atom_types(num):
    cdef int Natoms = len(num)
    cdef list atomic_types = sorted(list(set(num)))
    cdef int Ntypes = len(atomic_types)
    cdef list num_converted = [0]*Natoms
    cdef int i, j
    for i in range(Natoms):
        for j in range(Ntypes):
            if num[i] == atomic_types[j]:
                num_converted[i] = j
    return num_converted


cpdef void test_bondtypes():
    num = [10,12,32,10,10,3]
    #num = convert_atom_types(num1)
    atomic_types = sorted(list(set(num)))
    Ntypes = len(atomic_types)
    atomic_count = {type:list(num).count(type) for type in atomic_types}
    type_converter = {}
    for i, type in enumerate(atomic_types):
        type_converter[type] = i
    Nelements_2body = 0
    bondtypes_2body = -np.ones((Ntypes, Ntypes)).astype(int)
    count = 0
    print(atomic_types)
    for tt1 in atomic_types:
        for tt2 in atomic_types:
            type1, type2 = tuple(sorted([tt1, tt2]))
            t1 = type_converter[type1]
            t2 = type_converter[type2]
            print(type1, type2)
            if type1 < type2:
                if bondtypes_2body[t1,t2] == -1:
                    bondtypes_2body[t1,t2] = count
                    Nelements_2body += 1
                    count += 1
            elif type1 == type2 and (atomic_count[type1] > 1):  # or sum(self.pbc) > 0):
                if bondtypes_2body[t1,t2] == -1:
                    bondtypes_2body[t1,t2] = count
                    Nelements_2body += 1
                    count += 1

    print(type_converter)
    print(bondtypes_2body)
    print(Nelements_2body)

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
