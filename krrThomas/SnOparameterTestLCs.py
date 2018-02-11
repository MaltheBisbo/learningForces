import numpy as np
from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from gaussComparator_cosdist import gaussComparator_cosdist
from eksponentialComparator import eksponentialComparator
import matplotlib.pyplot as plt
from krr_class_new import krr_class

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
import time
atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)
a0 = atoms[10]
print('Ndata:', Ndata)
view(a0)
Energies = np.array([a.get_potential_energy() for a in atoms])

Rc1 = 5
binwidth1 = 0.3
sigma1 = 0.4

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

use_angular = True
gamma = 0
eta = 5.0

# Points in learning curve
Npoints = 10
gamma_array = [0,1,2,3,5]
eta_array = [0.01, 50, 5.0, 2.0, 1.0]
# Array to save LC results
MAE_all = np.zeros((len(eta_array), Npoints))

for n in range(len(eta_array)):
    gamma = gamma_array[n]
    eta = eta_array[n]
    E = Energies.copy()
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)
    fingerprint0 = featureCalculator.get_features(a0)
    length_feature = len(fingerprint0)
    
    # Save file
    #filename = 'SnO_features/SnO_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.1f}_sigma1_{2:.1f}_gamma_{3:d}.txt'.format(Rc1, binwidth1, sigma1, gamma)
    filename = 'SnO_features/SnO_radialAngFeatures_gauss_Rc1_2_{0:d}_{1:d}_binwidth1_{2:.1f}_Nbins2_{3:d}_sigma1_2_{4:.1f}_{5:.2f}_gamma_{6:d}.txt'.format(Rc1, Rc2, binwidth1, Nbins2, sigma1, sigma2, gamma)
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

    if use_angular:
        fingerprints[:, 3*Nradial:] *= eta

    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    
    # Perform training with cross-validation
    np.random.seed(101)
    
    Npermutations = 10
    N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
    MAE = np.zeros((Npermutations, Npoints))
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    
    for k in range(Npermutations):
        print('training: {}/{}'.format(k, Npermutations))
        permutation = np.random.permutation(Ndata)
        E = E[permutation]
        fingerprints = fingerprints[permutation]
        
        for i, N in enumerate(N_array):
            Esub = E[:N]
            fingerprints_sub = fingerprints[:N]
            
            MAE_temp, params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=10, **GSkwargs)
            MAE[k, i] += MAE_temp
            #print('params:', params)
    MAE_mean = MAE.mean(axis=0)
    MAE_std = MAE.std(axis=0)
    MAE_all[n] = MAE_mean


plt.figure(1)
for i in range(len(eta_array)):
    gamma = gamma_array[i]
    eta = eta_array[i]
    plt.loglog(N_array, MAE_all[i], label='gamma={}, eta={}'.format(gamma, eta))
plt.legend()
plt.title('Rc1={0:d}, bw1={1:.1f}, sigma1={2:.1f} \nRc2={3:d}, Nbins2={4:d}, sigma2={5:.1f}'.format(Rc1, binwidth1, sigma1, Rc2, Nbins2, sigma2))
plt.xlabel('# training data')
plt.ylabel('Energy MAE [eV/structure]')
plt.show()
