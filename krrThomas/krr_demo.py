import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature import Angular_Fingerprint
from gaussComparator import gaussComparator
from krr_ase import krr_class as krr_ase

from ase.io import read
from ase.visualize import view

atoms = read('graphene_data/all_done.traj', index=':10')
E = [a.get_potential_energy() for a in atoms]
a0 = atoms[0]
Ndata = len(atoms)

# Parameters for featureCalculator
Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

use_angular = False
gamma = 1
eta = 50

# Initialixa featureCalculator
featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)

# Calculate features + feature-gradients
fingerprints = featureCalculator.get_featureMat(atoms)
fingerprint_grads = featureCalculator.get_all_featureGradients(atoms)

# Split into test and training set
fingerprints_train = fingerprints[:-2]
fingerprints_test = fingerprints[-2:]
E_train = E[:-2]
E_test = E[-2:]

# Set up KRR-model
comparator = gaussComparator()
krr = krr_ase(comparator=comparator, featureCalculator=featureCalculator)

# define hyperparameters and train model
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
MAE, params = krr.train(data_values=E_train, featureMat=fingerprints_train, add_new_data=False, k=5, **GSkwargs)

E_ML = [krr.predict_energy(fnew=f) for f in fingerprints_test]
print(E_test)
print(E_ML)

view(a0)

r = np.linspace(0, Rc1, len(fingerprints[0]))
plt.figure()
plt.plot(r, fingerprints[0])
plt.show()
