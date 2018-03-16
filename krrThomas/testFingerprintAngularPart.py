import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint

from ase import Atoms
from ase.visualize import view

N = 3
dim = 3

L = 2
d = 1
pbc = [1,1,0]

"""
#x = np.array([0.2*L, 0.7*L, d/2])
x = np.array([0.8*L, 0.2*L, d/2])
positions = x.reshape((-1, dim))
a = Atoms('H',
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

x = np.array([0.2*L, 0.5*L, d/2, 0.7*L, 0.5*L, d/2])
#x = np.array([0.2*L, 0.7*L, d/2, 0.8*L, 0.2*L, d/2])
positions = x.reshape((-1, dim))
a = Atoms('H2',
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
x = np.array([0.2*L, 0.7*L, d/2, 0.3*L, 0.2*L, d/2, 0.7*L, 0.9*L, 3*d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H']*2 + ['He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""
x = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 1.5, 1, 0])
positions = x.reshape((-1,dim))
a = Atoms('H4',
          positions=positions,
          cell=[4,2,1],
          pbc=[0, 0, 0])
"""
#view(a)

    
Rc1 = 4
binwidth1 = 0.1
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.1

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=True)
fingerprint = featureCalculator.get_features(a)

print(featureCalculator.bondtypes_3body)
r_array = np.arange(len(fingerprint))

plt.figure(1)
plt.plot(r_array, fingerprint)
plt.show()

