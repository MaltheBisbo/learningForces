import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cpdef double test(int N):
    cdef np.ndarray[np.double_t, ndim=2] pos
    pos = np.random.rand(N,2)
    cdef double Rij = 0
    cdef np.ndarray[np.double_t, ndim=2] diffVec
    diffVec = np.zeros(3, dtype=np.double)
    cdef int i, j
    for i in range(N):
        for j in range(N):
            Rij += (pos[i,0] - pos[j,0])*(pos[i,0] - pos[j,0]) + (pos[i,1] - pos[j,1])*(pos[i,1] - pos[j,1]) + (pos[i,2] - pos[j,2])*(pos[i,2] - pos[j,2])
    return Rij
