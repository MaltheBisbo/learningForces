import numpy as np
from vector_krr_ase import vector_krr_class

from angular_fingerprintFeature_test3 import Angular_Fingerprint
from gaussComparator import gaussComparator
from vector_krr_ase import vector_krr_class

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb


atoms = read('graphene_data/graphene_all2.traj', index='100:120')
Natoms= len(atoms)
a0 = atoms[0]

# Get forces
F = np.array([a.get_forces() for a in atoms])
F = F.reshape((Natoms, -1))
print(F.shape)

# Setting up the featureCalculator
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


# Set up KRR model
comparator = gaussComparator()
vector_krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

# Training
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(-2,2,30)}
#GSkwargs = {'reg': [1e-5], 'sigma': [100]}
MAE, params = vector_krr.train(atoms_list=atoms, forces=F, add_new_data=False, **GSkwargs)
print(MAE, params)
