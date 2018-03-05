import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_test3 import Angular_Fingerprint
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint as Angular_Fingerprint_test
import time
import pdb

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

dim = 3

L = 2
d = 1
pbc = [1,1,1]

"""
#x = np.array([0.2*L, 0.7*L, d/2])
x = np.array([0.8*L, 0.2*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

N = 2
x = np.array([0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

N = 3
x = np.array([0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2,])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'He', 'He']
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

N = 4
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

N = 5
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2,
              0.9*L, 0.1*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""

atoms = read('graphene_data/graphene_all2.traj', index=':')
a = atoms[0]
atomtypes = a.get_atomic_numbers()
N = len(a.get_atomic_numbers())
x = a.get_positions().reshape(-1)

view(a)


Rc1 = 4
binwidth1 = 0.1
sigma1 = 0.2

Rc2 = 3
Nbins2 = 50
sigma2 = 0.2
use_angular = True

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=use_angular)

t0_feature1 = time.time()
fingerprint = featureCalculator.get_feature(a)
t1_feature1 = time.time()

t0_grad1 = time.time()
grad_fingerprint = featureCalculator.get_featureGradient(a)
t1_grad1 = time.time()



featureCalculator_test = Angular_Fingerprint_test(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=use_angular)

t0_feature2 = time.time()
fingerprint_test = featureCalculator_test.get_feature(a)
t1_feature2 = time.time()

t0_grad2 = time.time()
grad_fingerprint_test = featureCalculator_test.get_featureGradient(a)
t1_grad2 = time.time()

print('Calculation time - feature1:', t1_feature1 - t0_feature1)
print('Calculation time - feature2:', t1_feature2 - t0_feature2)
print('Calculation time - gradient1:', t1_grad1 - t0_grad1)
print('Calculation time - gradient2:', t1_grad2 - t0_grad2)

plt.figure(1)
plt.plot(np.arange(len(fingerprint_test))*binwidth1+binwidth1/2, fingerprint)
plt.plot(np.arange(len(fingerprint_test))*binwidth1+binwidth1/2, fingerprint_test, linestyle=':', color='k')

plt.figure(2)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, grad_fingerprint.T)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, grad_fingerprint_test.T, linestyle=':', color='k')

plt.figure(3)
plt.plot(np.arange(len(fingerprint_test))*binwidth1+binwidth1/2, fingerprint - fingerprint_test)

plt.figure(4)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, grad_fingerprint.T - grad_fingerprint_test.T)
plt.show()
