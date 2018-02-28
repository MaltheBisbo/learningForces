import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint
import time
import pdb

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

dim = 3

L = 2
d = 1
pbc = [1,1,1]

N = 4
x1 = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L-0.000001/2, d/2,
              0.7*L, 0.5*L, d/2])
"""
N = 5
x1 = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L-0.000001/2, 0.5*L, d/2,
              0.9*L, 0.1*L, d/2])
"""
positions1 = x1.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He']
a1 = Atoms(atomtypes,
           positions=positions1,
           cell=[L,L,d],
           pbc=pbc)


N = 4
x2 = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L+0.000001/2, d/2,
              0.7*L, 0.5*L, d/2])
"""
N = 5
x2 = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L+0.000001/2, 0.5*L, d/2,
              0.9*L, 0.1*L, d/2])
"""
positions2 = x2.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He']
a2 = Atoms(atomtypes,
           positions=positions2,
           cell=[L,L,d],
           pbc=pbc)
    
Rc1 = 1.6
binwidth1 = 0.6
sigma1 = 0.2

Rc2 = 3
Nbins2 = 50
sigma2 = 0.2

featureCalculator = Angular_Fingerprint(a1, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, nsigma=5, gamma=1, use_angular=False)
print('first 3')
fingerprint1 = featureCalculator.get_feature(a1)
print('second 3')
fingerprint2 = featureCalculator.get_feature(a2)

Rc1 = 1.59999
featureCalculator2 = Angular_Fingerprint(a1, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=1, use_angular=False)
print('first 2.99999')
fingerprint1_new = featureCalculator2.get_feature(a1)
print('second 2.99999')
fingerprint2_new = featureCalculator2.get_feature(a2)

#print(fingerprint2 - fingerprint1)

plt.figure(1)
plt.plot(np.arange(len(fingerprint1))*binwidth1+binwidth1/2, fingerprint1)
plt.plot(np.arange(len(fingerprint1))*binwidth1+binwidth1/2, fingerprint2, linestyle=':')

plt.figure(2)
plt.plot(np.arange(len(fingerprint1))*binwidth1+binwidth1/2, fingerprint1 - fingerprint2)
plt.plot(np.arange(len(fingerprint1_new))*binwidth1+binwidth1/2, fingerprint1_new - fingerprint2_new, linestyle=':')

plt.figure(3)
plt.plot(np.arange(len(fingerprint1))*binwidth1+binwidth1/2, fingerprint1 - fingerprint1_new)
plt.plot(np.arange(len(fingerprint1_new))*binwidth1+binwidth1/2, fingerprint2 - fingerprint2_new, linestyle=':')
plt.show()

