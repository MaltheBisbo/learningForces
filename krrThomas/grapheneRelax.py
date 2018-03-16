import numpy as np
import matplotlib.pyplot as plt
from krr_ase import krr_class as KRR
from angular_fingerprintFeature_test import Angular_Fingerprint
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from scipy.signal import argrelextrema
from custom_calculators import krr_calculator
from krr_calculator_no_z import krr_calculator as krr_calculator_no_z
import time

from ase import Atoms
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminBFGS
from ase.io import read, write
from ase.visualize import view

def main():
    atoms = read('graphene_data/graphene_all2.traj', index=':')
    E = [a.get_potential_energy() for a in atoms]
    
    a0 = atoms[0]
    Ndata = len(atoms)
    Nrelaxations = int(Ndata/4)
    print(Nrelaxations)
    
    # Split data into training and testing
    atoms_train = atoms[0:(Nrelaxations - 10)*4]
    atoms_test = atoms[(Nrelaxations - 10)*4:]
    atoms_test_start = atoms[(Nrelaxations - 10)*4::4]
    atoms_test_relaxed = atoms[(Nrelaxations - 10)*4 + 3::4]
    print(len(atoms))
    print(len(atoms_train))
    print(len(atoms_test))
    
    write('grapheneMLrelax/graphene_test_start.traj', atoms_test_start)
    write('grapheneMLrelax/graphene_test_relaxed.traj', atoms_test_relaxed)

    E_train = [a.get_potential_energy() for a in atoms_train]
    E_test = [a.get_potential_energy() for a in atoms_test]
    E_test_start = [a.get_potential_energy() for a in atoms_test_start]
    E_test_relaxed = [a.get_potential_energy() for a in atoms_test_relaxed]

    # angular fingerprint parameters
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2

    use_angular = False
    gamma = 1
    eta = 20

    # Initialize featureCalculator
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
    fingerprints = featureCalculator.get_featureMat(atoms, show_progress=True)

    # Initialize comparator and KRR model
    comparator = gaussComparator()
    krr = KRR(comparator=comparator, featureCalculator=featureCalculator)

    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    MAE, params = krr.train(data_values=E, featureMat=fingerprints, add_new_data=False, k=10, **GSkwargs)
    print('params:', params)
    print('MAE_energy: ', MAE)

    calculator = krr_calculator_no_z(krr)

    E_test_MLrelaxed = []
    atoms_test_MLrelaxed = []
    for i, a in enumerate(atoms_test_relaxed):
        a.set_calculator(calculator)
        dyn = BFGS(a, trajectory='grapheneMLrelax/grapheneNoZ_remove{}.traj'.format(i))
        dyn.run(fmax=0.1)
        atoms_test_MLrelaxed.append(a)
        E_test_MLrelaxed.append(krr.predict_energy(a))

    #write('grapheneMLrelax/grapheneAng_MLrelaxed.traj', atoms_test_MLrelaxed)
    #energies = np.array([E_test_start, E_test_relaxed, E_test_MLrelaxed])
    #energy_diff = np.array(E_test_relaxed) - np.array(E_test_MLrelaxed)
    #np.savetxt('grapheneMLrelax/grapheneAng_MLrelaxed_E.txt', energies, delimiter='\t')
    #print('Energy differende:\n', energy_diff)
    
    
if __name__ == "__main__":
    main()
