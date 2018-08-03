from doubleLJ import doubleLJ_energy
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0, 0])

Npoints = 1000
r = np.linspace(0.84, 3.5, Npoints)
X = np.c_[np.zeros((Npoints,3)), r]

eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
E = np.array([doubleLJ_energy(x, eps, r0, sigma) for x in X])

fs = 16
plt.plot(r, E, lw=2.5)
plt.xticks(np.arange(1,3.0,step=0.5), fontsize=fs)
plt.yticks(np.arange(-3,2,step=1), fontsize=fs)
plt.xlim([0.85, 2.0])
plt.ylim([-3,1.5])
plt.title('Lennard-Jones interaction', fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('E', fontsize=fs)
plt.savefig('doubleLJ.pdf')
plt.show()
