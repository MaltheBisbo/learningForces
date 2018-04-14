from doubleLJ import doubleLJ_energy
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0, 0])

Npoints = 100
r = np.linspace(0.85, 3.5, Npoints)
X = np.c_[np.zeros((Npoints,3)), r]

eps, r0, sigma = 1.0, 1.8, np.sqrt(0.3)
E = np.array([doubleLJ_energy(x, eps, r0, sigma) for x in X])

plt.plot(r, E)
plt.xlabel('r')
plt.ylabel('E')
plt.show()
