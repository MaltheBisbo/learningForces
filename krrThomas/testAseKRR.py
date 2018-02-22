import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from krr_class_new import krr_class


from ase import Atoms
from ase.io import read, write
from ase.visualize import view

atoms = read('graphene_data/all_done.traj', index=':')
E = [a.get_potential_energy() for a in atoms]
a0 = atoms[10]
Ndata = len(atoms)

Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

use_angular = False
gamma = 1
eta = 50

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)
fingerprint0 = featureCalculator.get_features(a0)
length_feature = len(fingerprint0)

# Filename
if not use_angular:
    filename = 'graphene_features/graphene_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}_gamma_{3:d}.txt'.format(Rc1, binwidth1, sigma1, gamma)
else:
    filename = 'graphene_features/graphene_radialAngFeatures_gauss_Rc1_2_{0:d}_{1:d}_binwidth1_{2:.1f}_Nbins2_{3:d}_sigma1_2_{4:.1f}_{5:.2f}_gamma_{6:d}.txt'.format(Rc1, Rc2, binwidth1, Nbins2, sigma1, sigma2, gamma)
    
# Load or calculate features
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

N_LCpoints = 10
N_array = np.logspace(1, np.log10(Ndata), N_LCpoints).astype(int)
FVU = np.zeros(N_LCpoints)
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
for i, N in enumerate(N_array):
    Esub = E[:N]
    fingerprints_sub = fingerprints[:N]
    
    FVU_temp, params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=10, **GSkwargs)
    FVU[i] = FVU_temp


plt.figure()
plt.loglog(N_array, FVU)
plt.show()

















