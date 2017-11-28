import numpy as np
from angular_fingerprintFeature2 import Angular_Fingerprint
from gaussComparator import gaussComparator
from angular_fingerprintFeature import Angular_Fingerprint as angular_fingerprint_tho
import matplotlib.pyplot as plt
from krr_class_new import krr_class

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[0]
view(a0)
print('Ndata:', Ndata)

E = np.array([a.get_potential_energy() for a in atoms])

Rc1 = 10
Rc2 = 5
binwidth1 = 0.1
sigma1 = 0.2
sigma2 = 0.1
featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, sigma1=sigma1, sigma2=sigma2, use_angular=False)


fingerprint0 = featureCalculator.get_features(a0)
length_feature = len(fingerprint0)

try:
    fingerprints = np.loadtxt('SnO_features/SnO_radialFeatures_Rc1_{}_binwidth1_{}_sigma1_{}.txt'.format(Rc1, binwidth1, sigma1), delimiter='\t')
except IOError:
    fingerprints = np.zeros((Ndata, length_feature))
    for i, structure in enumerate(atoms):
        print('calculating features: {}/{}\r'.format(i, Ndata), end='')
        fingerprints[i] = featureCalculator.get_features(structure)
    np.savetxt('SnO_features/SnO_radialFeatures_Rc1_{}_binwidth1_{}_sigma1_{}.txt'.format(Rc1, binwidth1, sigma1),
               fingerprints, delimiter='\t')

print('\n',fingerprints.shape)

comparator = gaussComparator()
krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

np.random.seed(101)
permutation = np.random.permutation(Ndata)
E = E[permutation]
fingerprints = fingerprints[permutation]

Npoints = 10
N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
FVU = np.zeros(Npoints)
GSkwargs = {'reg': np.logspace(-2, -7, 10), 'sigma': np.logspace(0,2,10)}
for i, N in enumerate(N_array):
    Esub = E[:N]
    fingerprints_sub = fingerprints[:N]
    
    #FVU_energy_array[i], FVU_force_array[i, :] = krr.train(Esub, Fsub, Gsub, Xsub, reg=reg)
    FVU[i], params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=10, **GSkwargs)
    print('params:', params)
    print('FVU_energy: {}\n'.format(FVU[i]))

plt.figure(1)
plt.loglog(N_array, FVU)
plt.ylim([10**-1, 10**1])

plt.figure(2)
plt.plot(np.arange(Ndata), E)
plt.show()
    
    



