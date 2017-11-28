import numpy as np
from angular_fingerprintFeature2 import Angular_Fingerprint
from angular_fingerprintFeature import Angular_Fingerprint as angular_fingerprint_tho
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[0]
print('Ndata:', Ndata)

Rc1 = 6
Rc2 = 5
binwidth1 = 0.1
sigma1 = 0.2
sigma2 = 0.1
featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, sigma1=sigma1, sigma2=sigma2, use_angular=False)

fingerprint0 = featureCalculator.get_features(a0)
length_feature - len(fingerprint0)

fingerprints = np.array([featureCalculator.get_feature(a) for a in atoms])
fingerprints = np.zeros((Ndata, length_fingerprint))
for i, structure in enumerate(atoms):
    print('calculating features: {}/{}'.format(i, Ndata))
    fingerprints[i] = featureCalculator.get_features(structure)

np.savetxt('SnO_radialFeatures_Rc1_{}_binwidth1_{}_sigma1_{}.txt'.format(Rc1, binwidth1, sigma1), fingerprints, delimiter='\t')




