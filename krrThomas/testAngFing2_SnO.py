import numpy as np
from angular_fingerprintFeature2 import Angular_Fingerprint
from gaussComparator import gaussComparator
from eksponentialComparator import eksponentialComparator
from angular_fingerprintFeature import Angular_Fingerprint as Angular_Fingerprint_tho
import matplotlib.pyplot as plt
from krr_class_new import krr_class

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
import time
atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[10]
view(a0)
print('Ndata:', Ndata)

E = np.array([a.get_potential_energy() for a in atoms])

Rc1 = 5
Rc2 = 5
binwidth1 = 0.1
sigma1 = 0.5
sigma2 = 0.1

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, sigma1=sigma1, sigma2=sigma2, use_angular=False)
fingerprint0 = featureCalculator.get_features(a0)
length_feature = len(fingerprint0)

featureCalculator_tho = Angular_Fingerprint_tho(a0, Rc=Rc1, binwidth1=binwidth1, binwidth2=0.5, sigma1=sigma1, sigma2=0.05)
res0_tho = featureCalculator_tho.get_features(a0)
keys_2body = list(res0_tho.keys())[:3]
print(keys_2body)
Nbins1 = int(Rc1/binwidth1)
Nelements = len(keys_2body) * Nbins1
fingerprint0_tho = np.zeros(Nelements)
for i, key in enumerate(keys_2body):
    fingerprint0_tho[i*Nbins1 : (i+1)*Nbins1] = res0_tho[key]

plt.figure(3)
plt.plot(np.arange(len(fingerprint0))*binwidth1, fingerprint0, label="new")
plt.plot(np.arange(len(fingerprint0_tho))*binwidth1, (fingerprint0_tho + 1)*sum(fingerprint0)/sum(fingerprint0_tho + 1), label="old")
plt.legend()

# Save file
filename = 'SnO_features/SnO_radialFeatures_gauss_Rc1_{}_binwidth1_{}_sigma1_{}.txt'.format(Rc1, binwidth1, sigma1)
try:
    fingerprints = np.loadtxt(filename, delimiter='\t')
except IOError:
    fingerprints = np.zeros((Ndata, length_feature))
    for i, structure in enumerate(atoms):
        print('calculating features: {}/{}\r'.format(i, Ndata), end='')
        fingerprints[i] = featureCalculator.get_features(structure)
    np.savetxt(filename, fingerprints, delimiter='\t')

# Set up KRR-model
comparator = gaussComparator()
krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

# Perform training with cross-validation
np.random.seed(101)
Npoints = 10
Npermutations = 2
N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
FVU = np.zeros((Npermutations, Npoints))
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}

for k in range(Npermutations):
    print('training: {}/{}'.format(k, Npermutations))
    permutation = np.random.permutation(Ndata)
    E = E[permutation]
    fingerprints = fingerprints[permutation]

    for i, N in enumerate(N_array):
        Esub = E[:N]
        fingerprints_sub = fingerprints[:N]
        
        FVU_temp, params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=10, **GSkwargs)
        FVU[k, i] += FVU_temp
        print('params:', params)
FVU_mean = FVU.mean(axis=0)
FVU_std = FVU.std(axis=0)
print(FVU_mean)
print(FVU_std)

result = np.r_[Rc1, binwidth1, sigma1, FVU_mean, FVU_std]

plt.figure(1)
plt.loglog(N_array, FVU_mean)
plt.ylim([10**-1, 10**1])

plt.figure(2)
plt.plot(np.arange(len(fingerprints[0]))*binwidth1, np.c_[fingerprints[0], fingerprints[10], fingerprints[20], fingerprints[100]])
plt.show()

    



