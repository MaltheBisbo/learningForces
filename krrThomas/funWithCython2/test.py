import numpy as np

def test(N):
    pos = np.random.rand(N,3)
    Rij = 0
    for i in range(N):
        for j in range(N):
            Rij += np.linalg.norm(pos[i]-pos[j])
    return Rij
