import numpy as np
import matplotlib.pyplot as plt
from krr_ase import krr_class as KRR
from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from scipy.signal import argrelextrema
from krr_calculator import krr_calculator
import time

from ase import Atoms
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminBFGS
from ase.io import read, write

def main():
    atoms = read('graphene_data/graphene_all2.traj', index=':')

    a0 = atoms[0]
    Ndata = len(atoms)
    Nrelaxations = int(Ndata/4)
    print(Nrelaxations)
    
    # Split data into training and testing
    atoms_train = atoms[0:(Nrelaxations - 10)*3]
    atoms_test = atoms[(Nrelaxations - 10)*3:]
    atoms_test_start = atoms[(Nrelaxations - 10)*3::4]
    atoms_test_target = atoms[(Nrelaxations - 10)*3 + 3::4]
    
    E_train = [a.get_potential_energy() for a in atoms_train]
    E_test = [a.get_potential_energy() for a in atoms_test]
    E_test_start = [a.get_potential_energy() for a in atoms_test_start]
    E_test_target = [a.get_potential_energy() for a in atoms_test_target]

    # angular fingerprint parameters
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2

    use_angular = False
    gamma = 1
    eta = 50

    # Initialize featureCalculator
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)
    fingerprints_train = []
    for i, a in enumerate(atoms_train):
        print('Calculating training features {}/{}\r'.format(i, len(atoms_train)), end='')
        fingerprints_train.append(featureCalculator.get_feature(a))
    print('\n')
    fingerprints_train = np.array(fingerprints_train)
    print(fingerprints_train.shape)

    # Initialize comparator and KRR model
    comparator = gaussComparator()
    krr = KRR(comparator=comparator, featureCalculator=featureCalculator)

    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    MAE, params = krr.train(data_values=E_train, featureMat=fingerprints_train, add_new_data=False, k=10, **GSkwargs)
    print('params:', params)
    print('MAE_energy: ', MAE)

    label = 'grapheneMLrelax/grapheneML'
    calculator = krr_calculator(krr, label)
    
    a = atoms_test_start[0]
    a.set_calculator(calculator)
    #dyn = SciPyFminBFGS(a, trajectory='grapheneMLrelax/graphene1.traj')
    dyn = BFGS(a, trajectory='grapheneMLrelax/graphene1.traj')
    dyn.run(fmax=0.1, n=200)

if __name__ == "__main__":
    main()
