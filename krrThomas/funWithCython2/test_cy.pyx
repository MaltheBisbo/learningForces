import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cpdef double test(int N):
    cdef np.ndarray[np.double_t, ndim=2] pos
    pos = np.random.rand(N,2)
    #cdef list pos = np.random.rand(N,2).tolist()
    cdef double Rij = 0

    cdef int i, j
    for i in range(N):
        for j in range(N):
            Rij += (pos[i,0] - pos[j,0])*(pos[i,0] - pos[j,0]) + (pos[i,1] - pos[j,1])*(pos[i,1] - pos[j,1]) + (pos[i,2] - pos[j,2])*(pos[i,2] - pos[j,2])
    return Rij
