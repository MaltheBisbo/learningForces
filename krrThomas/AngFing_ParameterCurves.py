import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from krr_class_new import krr_class

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
import time
atoms = read('fromThomas/data_SnO.traj', index=':')
Ndata = len(atoms)

E = np.array([a.get_potential_energy() for a in atoms])

Rc1 = 5
Rc2 = 5
binwidth1 = 0.1
Nbins2 = 30
sigma1 = 0.4
gamma = 3


def FVU_train(fingerprints, E, krr_model, Npoints, Npermutations):
    # Perform training with cross-validation
    np.random.seed(101)
    N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
    FVU = np.zeros((Npermutations, Npoints))
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
    FVU_mean = FVU.mean(axis=0)
    return FVU_mean[-1]

Neta = 15
eta_array = np.linspace(1, 30, Neta).astype(int)
results = []

plt.figure(1)
for name in ['', '_r_fcut', '_fcut']:
    for sigma2 in [0.05, 0.1, 0.2]:
        filename = 'SnO_features/SnO_radialAngFeatures_gauss{7:s}_Rc1_2_{0:d}_{1:d}_binwidth1_{2:.1f}_Nbins2_{3:d}_sigma1_2_{4:.1f}_{5:.2f}_gamma_{6:d}.txt'.format(Rc1, Rc2, binwidth1, Nbins2, sigma1, sigma2, gamma, name)
        fingerprints = np.loadtxt(filename, delimiter='\t')

        print(sigma2)
        # Set up KRR-model
        featureCalculator = Angular_Fingerprint(atoms[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=True)
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
        
        MAEcurve = np.zeros(Neta)
        for i, eta in enumerate(eta_array):
            Nradial = int(Rc1/binwidth1)
            fingerprints_eta = fingerprints.copy()
            fingerprints_eta[:, 3*Nradial:] *= eta
            MAEcurve[i] = FVU_train(fingerprints_eta, E, krr, Npoints=10, Npermutations=5)
            print(MAEcurve)
        plt.plot(eta_array, MAEcurve, label='{0:s} sigmaAng={1:.2f}'.format(name, sigma2))
        results.append(MAEcurve)

results = np.array(results)
np.savetxt('resultsAngFing_paramCurves2.txt', results, delimiter='\t')

plt.legend()
plt.show()
