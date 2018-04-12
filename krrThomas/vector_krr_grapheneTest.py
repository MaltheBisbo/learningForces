import numpy as np
from vector_krr_ase2 import vector_krr_class

from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from gaussComparator2 import gaussComparator
from maternComparator import maternComparator

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb


atoms = read('graphene_data/graphene_all2.traj', index=':')
atoms = atoms[0::2]
atoms = atoms[:40]
#atoms = read('graphene_data/all_done.traj', index='100:110')
Natoms = len(atoms)
a0 = atoms[0]
view(atoms)

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
eta = 20

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)


# Set up KRR model
comparator = maternComparator()
vector_krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

# Training
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,4,30)}
#GSkwargs = {'reg': [1e-5], 'sigma': [5]}
MAE, params = vector_krr.train(atoms_list=atoms, forces=F, add_new_data=False, k=5, **GSkwargs)
print(MAE, params)






"""
comparator.set_args(sigma=2)
featureMat = vector_krr.featureMat
similarityMat = comparator.get_similarity_matrix(featureMat)[:Natoms,:][:,:Natoms]
print(similarityMat.shape)
np.set_printoptions(precision=2, threshold=10000, linewidth=150)
print(similarityMat)
"""
