import numpy as np
from scipy.spatial.distance import cosine

from angular_fingerprintFeature_test3 import Angular_Fingerprint
from gaussComparator import gaussComparator
from maternComparator import maternComparator
from vector_krr_ase2 import vector_krr_class

import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb

def LC(atoms, featureCalculator, Ntrain_array):
    Ndata = len(atoms)
    permut = np.random.permutation(Ndata).astype(int)
    atoms = [atoms[i] for i in permut]
    
    E = np.array([a.get_potential_energy() for a in atoms])
    F = np.array([a.get_forces() for a in atoms])
    
    # Set up KRR-model
    comparator = maternComparator()
    krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,4,20)}

    MAE_array = np.zeros(len(Ntrain_array))
    for i, Ntrain in enumerate(Ntrain_array):
        atoms_sub = atoms[:Ntrain]
        F_sub = F[:Ntrain]
        MAE, params = krr.train(atoms_list=atoms_sub,
                                forces=F_sub,
                                add_new_data=False,
                                k=5,
                                **GSkwargs)
        MAE_array[i] = MAE
        print('MAE:', MAE)
        print(params)
        
    return MAE_array


if __name__ == '__main__':
    atoms = read('graphene_data/graphene_all2.traj', index=':')
    atoms = atoms[0::2]
    atoms = atoms[:40]
    a0 = atoms[0]
    Ndata = len(atoms)
    
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

    # Ntrain_array
    Npoints = 10
    Ntrain_array = np.logspace(1,np.log10(50), Npoints).astype(int)

    MAE_array = LC(atoms, featureCalculator, Ntrain_array=Ntrain_array)

    plt.figure()
    plt.plot(Ntrain_array, MAE_array)
    plt.show()

    
    
