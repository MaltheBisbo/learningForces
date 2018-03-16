import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint
from angular_fingerprintFeature_test import Angular_Fingerprint as Angular_Fingerprint_test
import time
import pdb

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
a = atoms[3]
atomtypes = a.get_atomic_numbers()
N = len(a.get_atomic_numbers())
x = a.get_positions().reshape(-1)

view(a)

dx = 0.000001
    
Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 50
use_angular = True

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
fingerprint = featureCalculator.get_feature(a)

t0_analytic = time.time()
grad_fingerprint = featureCalculator.get_featureGradient(a)
t1_analytic = time.time()

t0_numeric = time.time()
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

r_array = np.arange(len(fingerprint))*binwidth1+binwidth1/2

plt.figure(10)
plt.plot(np.arange(len(fingerprint))*binwidth1+binwidth1/2, fingerprint)

plt.figure(11)
plt.plot(np.arange(len(fingerprint))*binwidth1, num_grad_fingerprint.T)
plt.plot(np.arange(len(fingerprint))*binwidth1, grad_fingerprint, linestyle=':', color='k')

plt.figure(12)
plt.plot(np.arange(len(fingerprint))*binwidth1, num_grad_fingerprint.T - grad_fingerprint)
plt.show()
"""
featureCalculator_test = Angular_Fingerprint_test(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, nsigma=5, gamma=gamma, use_angular=use_angular)
fingerprint_test = featureCalculator_test.get_feature(a)

t0_analytic_test = time.time()
grad_fingerprint_test = featureCalculator_test.get_featureGradient(a)
t1_analytic_test = time.time()

t0_numeric_test = time.time()
num_grad_fingerprint_test = np.zeros((N*dim, len(fingerprint_test)))
for i in range(N*dim):
    x_pertub = np.zeros(N*dim)
    x_pertub[i] = dx
    pos_down = np.reshape(x - x_pertub/2, (-1, dim))
    pos_up = np.reshape(x + x_pertub/2, (-1, dim))

    a_down = Atoms(atomtypes, positions=pos_down, cell=[L,L,d], pbc=pbc)
    a_up = Atoms(atomtypes, positions=pos_up, cell=[L,L,d], pbc=pbc)
    fingerprint_down = featureCalculator_test.get_feature(a_down)
    fingerprint_up = featureCalculator_test.get_feature(a_up)
    num_grad_fingerprint_test[i] = (fingerprint_up - fingerprint_down)/dx
t1_numeric_test = time.time()

print('Calculation time - analytic:', t1_analytic_test - t0_analytic_test)
print('Calculation time - numeric:', t1_numeric_test - t0_numeric_test)

print(featureCalculator_test.bondtypes_2body)
print(featureCalculator_test.bondtypes_3body)

r_array = np.arange(len(fingerprint_test))*binwidth1+binwidth1/2

plt.figure(20)
plt.plot(np.arange(len(fingerprint_test))*binwidth1+binwidth1/2, fingerprint_test)

plt.figure(21)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, num_grad_fingerprint_test.T)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, grad_fingerprint_test, linestyle=':', color='k')

plt.figure(22)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, num_grad_fingerprint_test.T - grad_fingerprint_test)

plt.figure(30)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, grad_fingerprint - grad_fingerprint_test)

plt.figure(31)
plt.plot(np.arange(len(fingerprint_test))*binwidth1, num_grad_fingerprint.T - num_grad_fingerprint_test.T)
plt.show()
"""
