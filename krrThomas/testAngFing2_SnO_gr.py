import numpy as np
from angular_fingerprintFeature2 import Angular_Fingerprint
from gaussComparator import gaussComparator
#from eksponentialComparator import eksponentialComparator
#from angular_fingerprintFeature import Angular_Fingerprint as Angular_Fingerprint_tho
from krr_class_new import krr_class
import time
import sys

from ase import Atoms
from ase.io import read, write
from ase.visualize import view


atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[10]

E = np.array([a.get_potential_energy() for a in atoms])

arg = int(sys.argv[1])
print('arg:', arg)
binwidth1_array = np.linspace(0.02, 0.2, 10)
sigma1_array = np.linspace(0.1, 1, 10)

Rc1 = 7
binwidth1 = 0.1  # binwidth1_array[arg // 10]
sigma1 = 0.4  # sigma1_array[arg % 10]

Rc2 = 5
Nbins2 = 30
sigma2 = 0.1
gamma = 3
eta = 5

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, use_angular=True)
fingerprint0 = featureCalculator.get_features(a0)
length_feature = len(fingerprint0)

# Save file
filename = 'SnO_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}.txt'.format(Rc1, binwidth1, sigma1)
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
Npermutations = 10
N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
FVU = np.zeros((Npermutations, Npoints))
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}

for k in range(Npermutations):
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

result = np.r_[Rc1, binwidth1, sigma1, FVU_mean, FVU_std]
np.savetxt('SnO_radialResults_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}.txt'.format(Rc1, binwidth1, sigma1), result, delimiter='\t')

    



