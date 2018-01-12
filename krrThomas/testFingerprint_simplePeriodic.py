import numpy as np
import matplotlib.pyplot as plt

from fingerprintFeature import fingerprintFeature
#from angular_fingerprintFeature2 import Angular_Fingerprint
from angular_fingerprintFeature_m import Angular_Fingerprint
from angular_fingerprintFeature import Angular_Fingerprint as Angular_Fingerprint_tho

from ase import Atoms
from ase.visualize import view

L=2
d = 1
a = Atoms('H2',
          positions=[[0.2*L, 0.7*L, d/2], [0.8*L, 0.2*L, d/2]],
          cell=[L,L,d],
          pbc=[1,0,0])
view(a)

Rc1 = 3
Rc2 = 1
binwidth1 = 0.05
sigma1 = 0.5
sigma2 = 0.1

#featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, sigma1=sigma1, sigma2=sigma2, use_angular=False)
featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=False)
fingerprint = featureCalculator.get_features(a)
length_feature = len(fingerprint)

featureCalculator_tho = Angular_Fingerprint_tho(a, Rc=Rc1, binwidth1=binwidth1, binwidth2=0.5, sigma1=sigma1, sigma2=0.05)
res_tho = featureCalculator_tho.get_features(a)
keys_2body = list(res_tho.keys())[:1]
print(keys_2body)
Nbins1 = int(Rc1/binwidth1)
Nelements = len(keys_2body) * Nbins1
fingerprint_tho = np.zeros(Nelements)
for i, key in enumerate(keys_2body):
    fingerprint_tho[i*Nbins1 : (i+1)*Nbins1] = res_tho[key]

fraction = 1  # sum(fingerprint) / sum(fingerprint_tho+1)
print('fraction:', fraction)
plt.figure(1)
plt.plot(np.arange(len(fingerprint))*binwidth1, fingerprint, label='new')
plt.plot(np.arange(len(fingerprint_tho))*binwidth1, (fingerprint_tho+1)*fraction, label='old')
plt.legend()
plt.show()
