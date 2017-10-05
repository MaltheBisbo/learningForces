import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='LC_bob_N7_forces.txt', delimiter='\t')
N = data[:,0]
MAE_energy = data[:,1]
MAE_force = data[:, 2:]

plt.figure(1)
plt.loglog(N, MAE_energy)
plt.xlabel('N')
plt.ylabel('MAE_energy')

plt.figure(2)
plt.loglog(N, MAE_force)
plt.xlabel('N')
plt.ylabel('MAE_force')

plt.show()
