import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint

from ase import Atoms
from ase.visualize import view

N = 3
dim = 3

L = 2
d = 1
pbc = [0,1,0]

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
atomtypes = ['H', 'H', 'He']
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
Nbins2 = 50
sigma2 = 0.2

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=True)
fingerprint = featureCalculator.get_features(a)
grad_fingerprint = featureCalculator.get_featureGradients(a)


dx = 0.00001
num_grad_fingerprint = np.zeros((N*dim, len(fingerprint)))
for i in range(N*dim):
    x_pertub = np.zeros(N*dim)
    x_pertub[i] = dx
    pos_down = np.reshape(x - x_pertub/2, (-1, dim))
    pos_up = np.reshape(x + x_pertub/2, (-1, dim))

    a_down = Atoms(atomtypes, positions=pos_down, cell=[L,L,d], pbc=pbc)
    a_up = Atoms(atomtypes, positions=pos_up, cell=[L,L,d], pbc=pbc)
    fingerprint_down = featureCalculator.get_features(a_down)
    fingerprint_up = featureCalculator.get_features(a_up)
    num_grad_fingerprint[i] = (fingerprint_up - fingerprint_down)/dx

#print(fingerprint)
#print(num_grad_fingerprint[0])
#print(grad_fingerprint)
print(featureCalculator.bondtypes_2body)
print(featureCalculator.bondtypes_3body)

r_array = np.arange(len(fingerprint))*binwidth1+binwidth1/2
print(np.trapz(fingerprint, r_array))

plt.figure(20)
plt.plot(np.arange(len(fingerprint))*binwidth1+binwidth1/2, fingerprint)

plt.figure(21)
plt.plot(np.arange(len(fingerprint))*binwidth1, num_grad_fingerprint.T)
plt.plot(np.arange(len(fingerprint))*binwidth1, grad_fingerprint, linestyle=':')
plt.show()

