import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint
import time

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

dim = 3

L = 2
d = 1
pbc = [0,0,0]

"""
#x = np.array([0.2*L, 0.7*L, d/2])
x = np.array([0.8*L, 0.2*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
N = 2
x = np.array([0.2*L, 0.5*L, d/2, 0.7*L, 0.5*L, d/2])
#x = np.array([0.2*L, 0.7*L, d/2, 0.8*L, 0.2*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['H', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
N = 3
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, 3*d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'H', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

x = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 1.5, 1, 0])
positions = x.reshape((-1,dim))
a = Atoms('H4',
          positions=positions,
          cell=[4,2,1],
          pbc=[0, 0, 0])


x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)


atoms = read('graphene_data/graphene_all2.traj', index=':')
a = atoms[0]
atomtypes = a.get_atomic_numbers()
N = len(a.get_atomic_numbers())
x = a.get_positions().reshape(-1)
"""
view(a)

    
Rc1 = 4
binwidth1 = 0.1
sigma1 = 0.2

Rc2 = 3
Nbins2 = 50
sigma2 = 0.2

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=False)
fingerprint = featureCalculator.get_feature(a)

t0_analytic = time.time()
grad_fingerprint = featureCalculator.get_featureGradient(a)
t1_analytic = time.time()

if np.any(np.isnan(grad_fingerprint)):
    print('ERROR: nan in feature gradient')

t0_numeric = time.time()
dx = 0.000001
num_grad_fingerprint = np.zeros((N*dim, len(fingerprint)))
for i in range(N*dim):
    x_pertub = np.zeros(N*dim)
    x_pertub[i] = dx
    pos_down = np.reshape(x - x_pertub/2, (-1, dim))
    pos_up = np.reshape(x + x_pertub/2, (-1, dim))

    a_down = Atoms(atomtypes, positions=pos_down, cell=[L,L,d], pbc=pbc)
    a_up = Atoms(atomtypes, positions=pos_up, cell=[L,L,d], pbc=pbc)
    fingerprint_down = featureCalculator.get_feature(a_down)
    fingerprint_up = featureCalculator.get_feature(a_up)
    num_grad_fingerprint[i] = (fingerprint_up - fingerprint_down)/dx
t1_numeric = time.time()

print('Calculation time - analytic:', t1_analytic - t0_analytic)
print('Calculation time - numeric:', t1_numeric - t0_numeric)

print(featureCalculator.bondtypes_2body)
print(featureCalculator.bondtypes_3body)

print('analytic:\n', grad_fingerprint[1])
print('numeric:\n', num_grad_fingerprint.T[1])
print('Difference:\n', (num_grad_fingerprint.T - grad_fingerprint)[1])

r_array = np.arange(len(fingerprint))*binwidth1+binwidth1/2
print(np.trapz(fingerprint, r_array))

plt.figure(20)
plt.plot(np.arange(len(fingerprint))*binwidth1+binwidth1/2, fingerprint)

plt.figure(21)
plt.plot(np.arange(len(fingerprint))*binwidth1, num_grad_fingerprint.T)
plt.plot(np.arange(len(fingerprint))*binwidth1, grad_fingerprint, linestyle=':', color='k')

plt.figure(22)
plt.plot(np.arange(len(fingerprint))*binwidth1, num_grad_fingerprint.T - grad_fingerprint)
plt.show()
