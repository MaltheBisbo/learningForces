import numpy as np
from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from eksponentialComparator import eksponentialComparator
from krr_class_new import krr_class

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
import time

import pdb

def LC(atoms, fingerprintCalculator, feature_filename, N_LCpoints=10, Npermutations=10):
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
    if use_angular:
        fingerprints[:, 3*Nradial:] *= eta
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
binwidth1 = 0.05
sigma1 = 0.1

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

use_angular = False
gamma = 1
eta = 50

binwidth1_array = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1])
sigma1_array = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
FVU_grid = np.zeros((len(binwidth1_array), len(sigma1_array)))

for i, binwidth1 in enumerate(binwidth1_array):
    for j, sigma1 in enumerate(sigma1_array):
        atoms = read('graphene_data/all_done.traj', index=':')
        a0 = atoms[10]
        
        featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)
        fingerprint0 = featureCalculator.get_features(a0)
        length_feature = len(fingerprint0)

        # Save file
        if not use_angular:
            filename = 'graphene_features/graphene_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}_gamma_{3:d}.txt'.format(Rc1, binwidth1, sigma1, gamma)
        else:
            filename = 'graphene_features/graphene_radialAngFeatures_gauss_Rc1_2_{0:d}_{1:d}_binwidth1_{2:.1f}_Nbins2_{3:d}_sigma1_2_{4:.1f}_{5:.2f}_gamma_{6:d}.txt'.format(Rc1, Rc2, binwidth1, Nbins2, sigma1, sigma2, gamma)

        FVU_mean, FVU_std, N_array, fingerprints = LC(atoms, featureCalculator, feature_filename=filename)

        FVU_grid[i,j] = FVU_mean[-1]

np.savetxt('grapheneFingParams/Rc1_5_bw1_sigma1_grid.txt', FVU_grid, delimiter='\t')
print(FVU_grid)

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(binwidth1_array, sigma1_array)

surf = ax.plot_surface(X, Y, FVU_grid.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.text2D(0.05, 0.96, 'graphene prediction MAE for Ntrain=137 \n using the radial fingerprint with Rc1=5', transform=ax.transAxes, size=14)
ax.set_xlabel('binwidth1')
ax.set_ylabel('sigma1')
ax.set_zlabel('MAE')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""
plt.figure(1)
plt.title('Learning curve - SnO (only angular part): \nRc2={0:d}, Nbins2={1:d}, sigma2={2:.1f}, gamma={3:d}'.format(Rc2, Nbins2, sigma2, gamma))
plt.loglog(N_array, FVU_mean)

plt.figure(2)
plt.title('Feature examples - SnO (only angular part): \nRc2={0:d}, Nbins2={1:d}, sigma2={2:.1f}, gamma={3:d}'.format(Rc2, Nbins2, sigma2, gamma))
plt.plot(np.arange(len(fingerprints[0])), np.c_[fingerprints[0], fingerprints[10], fingerprints[20], fingerprints[100]])
plt.show()
"""
