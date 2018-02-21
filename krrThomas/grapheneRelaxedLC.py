import numpy as np
from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from eksponentialComparator import eksponentialComparator
from krr_class_new import krr_class

import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
import time

import pdb

def LC(atoms, featureCalculator, feature_filename, N_LCpoints=10, Npermutations=10, use_angular=False, eta=1):
    Ndata = len(atoms)
    E = np.array([a.get_potential_energy() for a in atoms])
    
    try:
        fingerprints = np.loadtxt(filename, delimiter='\t')
    except IOError:
        fingerprints = np.zeros((Ndata, length_feature))
        for i, structure in enumerate(atoms):
            print('calculating features: {}/{}\r'.format(i, Ndata), end='')
            fingerprints[i] = featureCalculator.get_features(structure)
        np.savetxt(filename, fingerprints, delimiter='\t')
    print(filename)
    Nradial = int(np.ceil(Rc1/binwidth1))
    print('Nradiea:', Nradial)
    Nbondtypes_2body = len(featureCalculator.bondtypes_2body)
    if use_angular:
        fingerprints[:, Nbondtypes_2body*Nradial:] *= eta
    print('Nbins1:', featureCalculator.Nbins1)
    print(len(fingerprints[0]))
    print(featureCalculator.bondtypes_3body)
    
    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    
    # Perform training with cross-validation
    np.random.seed(101)
    N_LCpoints = 10
    Npermutations = 10
    N_array = np.logspace(1, np.log10(Ndata), N_LCpoints).astype(int)
    FVU = np.zeros((Npermutations, N_LCpoints))
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    
    for k in range(Npermutations):
        print('training: {}/{}'.format(k, Npermutations))
        permutation = np.random.permutation(Ndata)
        E = E[permutation]
        fingerprints = fingerprints[permutation]

        for i, N in enumerate(N_array):
            Esub = E[:N]
            fingerprints_sub = fingerprints[:N]
            
            FVU_temp, params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=10, **GSkwargs)
            FVU[k, i] += FVU_temp
            #print('params:', params)
    FVU_mean = FVU.mean(axis=0)
    FVU_std = FVU.std(axis=0)
    print(FVU_mean)
    print(FVU_std)
    
    return FVU_mean, FVU_std, N_array, fingerprints



Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

use_angular = True
gamma = 1
eta = 50

eta_array = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
N_LCpoints = 10
MAE = np.zeros((len(eta_array), N_LCpoints))

plt.figure(1)
plt.title('Feature examples - graphene: \nRc2={0:d}, Nbins2={1:d}, sigma2={2:.1f}, gamma={3:d}'.format(Rc2, Nbins2, sigma2, gamma))

for i, eta in enumerate(eta_array):
    atoms = read('graphene_data/graphene_all2.traj', index=':')
    a0 = atoms[10]
        
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)
    fingerprint0 = featureCalculator.get_features(a0)
    length_feature = len(fingerprint0)

    # Save file
    if not use_angular:
        filename = 'graphene_features/graphene_unrelaxed3_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}_gamma_{3:d}.txt'.format(Rc1, binwidth1, sigma1, gamma)
    else:
        filename = 'graphene_features/graphene_unrelaxed3_radialAngFeatures_gauss_Rc1_2_{0:d}_{1:d}_binwidth1_{2:.1f}_Nbins2_{3:d}_sigma1_2_{4:.1f}_{5:.2f}_gamma_{6:d}.txt'.format(Rc1, Rc2, binwidth1, Nbins2, sigma1, sigma2, gamma)

    MAE_mean, MAE_std, N_array, fingerprints = LC(atoms, featureCalculator, feature_filename=filename, N_LCpoints=N_LCpoints, use_angular=use_angular, eta=eta)
    
    MAE[i] = MAE_mean
    plt.plot(np.arange(len(fingerprints[0])), fingerprints[0])
    
np.savetxt('grapheneFingParams/angFing_unrelaxed3_Rc2_4_Nbins2_30_sigma2_0.2_gamma_1.txt', MAE, delimiter='\t')

plt.figure(2)
plt.title('Learning curve - graphene: \nRc2={0:d}, Nbins2={1:d}, sigma2={2:.1f}, gamma={3:d}'.format(Rc2, Nbins2, sigma2, gamma))
for i, eta in enumerate(eta_array):
    plt.loglog(N_array, MAE[i], label='eta={0}, MAE={1:.3f}'.format(eta, MAE[i,-1]))
plt.legend()
plt.xlabel('# training data')
plt.ylabel('MAE')

plt.savefig('grapheneFingParams/results/angLC_unrelaxed3_Rc2_{0:d}_Nbins2_{1:d}_sigma2_{2:.1f}_gamma_{3:d}.png'.format(Rc2, Nbins2, sigma2, gamma), bbox_inches='tight')
plt.show()

