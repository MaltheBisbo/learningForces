import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase2 import krr_class
from delta_functions.delta import delta as deltaFunc
from custom_calculators import krr_calculator

from doubleLJ import doubleLJ_energy_ase as E_doubleLJ
from delta_functions.delta import delta as deltaFunc

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.data import covalent_radii

#from helpFunctionsDLJ19 import get_structure, get_structure_list


a_data = read('data_N200_d1.traj', index=':')
E_data = np.array([E_doubleLJ(a) for a in a_data])
a0 = a_data[0]

# Set up featureCalculator
Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 30
use_angular = True

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

# Set up KRR-model
comparator = gaussComparator()
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator)
sigma = 10
#GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}
GSkwargs = {'reg': [1e-7], 'sigma': np.logspace(0,3,10)}


# Training data
a_train1 = a_data[:50]
a_train2 = a_data[50:100]
E_train1 = E_data[:50]
E_train2 = E_data[50:100]

# Test data
a_test = a_data[100:]
E_test = E_data[100:]


MAE1, params1 = krr.train(atoms_list=a_train1,
                          add_new_data=False,
                          k=2,
                          **GSkwargs)
print(MAE1, params1)

Epred1 = np.array([krr.predict_energy(a) for a in a_test])


MAE2, params2 = krr.train(atoms_list=a_train2,
                          add_new_data=True,
                          k=2,
                          **GSkwargs)
print(MAE2, params2)

Epred2 = np.array([krr.predict_energy(a) for a in a_test])

test_error1 = np.mean(np.abs(E_test - Epred1))
test_error2 = np.mean(np.abs(E_test - Epred2))

print(test_error1)
print(test_error2)
