import numpy as np
import matplotlib.pyplot as plt
from krr_class_new import krr_class
from angular_fingerprintFeature_m import Angular_Fingerprint
from gaussComparator import gaussComparator
from scipy.signal import argrelextrema
from krr_calculator import krr_calculator
import time

from ase import Atoms
from ase.optimize import BFGS
from ase.io import read, write
from ase.visualize import view

def loadTraj():
    atoms = read('work_folder/all.traj', index=':')
    atoms = atoms[0:15000]
    #atoms = atoms[::2]
    #atoms = atoms[:Ndata]
    Ndata = len(atoms)
    Na = 24
    dim = 3
    Ntraj = len(atoms)
    
    pos = np.zeros((Ntraj,Na,dim))
    E = np.zeros(Ntraj)
    F = np.zeros((Ntraj, Na, dim))
    for i, a in enumerate(atoms):
        print('Loading: {}/{}\r'.format(i, Ndata), end='')
        pos[i] = a.positions
        E[i] = a.get_potential_energy()
        F[i] = a.get_forces()

    return atoms, pos.reshape((Ntraj, Na*dim)), E, F.reshape((Ntraj, Na*dim))


def main():
    np.random.seed(100)
    atoms, pos, E, F = loadTraj()
    #atoms = read('work_folder/all.traj', index=':100')
    print('\n',len(atoms))
    a = atoms[0]

    Ntrain = 1000
    Ntest = 500
    Ntest2 = 500
    index = np.random.permutation(10000)
    i_train = index[:Ntrain].astype(int)
    i_test = (np.random.permutation(5000)+10000)[:Ntest].astype(int)
    i_test2 = index[Ntrain:Ntrain+Ntest2]
    
    Rc1 = 5
    binwidth1 = 0.1
    sigma1 = 0.4

    Rc2 = 3
    Nbins2 = 30
    sigma2 = 0.1
    
    gamma = 5
    
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=0, use_angular=False)

    fingerprints_train = []
    for num, i in enumerate(i_train):
        print('Training features: {}/{}\r'.format(num, Ntrain), end='')
        feature_i = featureCalculator.get_features(atoms[i])
        fingerprints_train.append(feature_i)
    fingerprints_train = np.array(fingerprints_train)
    print('\n')
    
    fingerprints_test = []
    for num, i in enumerate(i_test):
        print('Test features: {}/{}\r'.format(num, Ntest), end='')
        feature_i = featureCalculator.get_features(atoms[i])
        fingerprints_test.append(feature_i)
    fingerprints_test = np.array(fingerprints_test)
    print('\n')
    
    fingerprints_test2 = []
    for num, i in enumerate(i_test):
        print('Test2 features: {}/{}\r'.format(num, Ntest2), end='')
        feature_i = featureCalculator.get_features(atoms[i])
        fingerprints_test2.append(feature_i)
    fingerprints_test2 = np.array(fingerprints_test2)

    Etrain = E[i_train]
    Etest = E[i_test]
    Etest2 = E[i_test2]
    
    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    
    # Perform training with cross-validation
    Npoints = 30
    Ndata = Ntrain
    N_array = np.logspace(1, np.log10(Ndata), Npoints).astype(int)
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1,2,10)}

    MAE_val = np.zeros(Npoints)
    MAE_test = np.zeros(Npoints)
    MAE_test2 = np.zeros(Npoints)
    for i, N in enumerate(N_array):
        Esub = E[:N]
        fingerprints_sub = fingerprints_train[:N]
        
        MAE_val_temp, params = krr.train(Esub, featureMat=fingerprints_sub, add_new_data=False, k=5, **GSkwargs)
        MAE_val[i] = MAE_val_temp
        print('N:', N, 'params:', params)
        Epredict = np.array([krr.predict_energy(fnew=f) for f in fingerprints_test])
        Epredict2 = np.array([krr.predict_energy(fnew=f) for f in fingerprints_test2])
        MAE_test[i] = np.mean(np.abs(Epredict - Etest))
        MAE_test2[i] = np.mean(np.abs(Epredict2 - Etest2))

    print(MAE_val)
    print(MAE_test)
    print(MAE_test2)

    plt.figure()
    plt.loglog(N_array, MAE_test)
    plt.loglog(N_array, MAE_test2)
    
    
    plt.figure()
    plt.plot(np.arange(len(fingerprints_train[0]))*Rc1/len(fingerprints_train[0]), fingerprints_train[0], label='index 0')
    plt.plot(np.arange(len(fingerprints_train[0]))*Rc1/len(fingerprints_train[0]), fingerprints_train[40], label='index 40')
    plt.plot(np.arange(len(fingerprints_train[0]))*Rc1/len(fingerprints_train[0]), fingerprints_train[90], label='index 90')
    plt.legend()
    plt.show()
    """
    [ 19.15921531  19.86766022  21.21613115  15.96245691  16.91681297
  14.8915245   12.6001315   11.93777066  12.33840395  14.75289874
   9.30982123   8.98324929   7.39914261   6.48309714   6.09473549
   6.03771978   5.64555205   5.76005092   5.60891719   5.5894711
   5.5726956    5.40067667   5.37019041   5.33675678   5.52035752
   5.96919136   5.6995485    6.68662729   6.13550929   5.94384627]
[ 17.48593875  18.285844    20.43432449  14.74036565  15.77212828
  13.26720632  11.38015665  10.37589333  10.10622903  13.31456299
   7.87100404   7.47538609   6.44173639   5.68295358   5.44522854
   5.48219097   5.20928989   5.33421367   5.28745139   5.34007973
   5.314172     5.14872923   5.11827076   5.09639421   5.16014279
   5.45981102   5.4308997    6.03696307   5.63806243   5.52051774]
    """

    
    
if __name__ == "__main__":
    main()
