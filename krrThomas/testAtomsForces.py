import numpy as np
from scipy.spatial.distance import cosine

from angular_fingerprintFeature_test3 import Angular_Fingerprint
from gaussComparator import gaussComparator
from krr_ase import krr_class

import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb

def predictEandF(atoms, featureCalculator, feature_filename, feature_grad_filename, Nsplit=5, eta=1):
    Ndata = len(atoms)
    Natoms = atoms[0].get_number_of_atoms()
    dim = 3
    
    E = np.array([a.get_potential_energy() for a in atoms])
    F = np.array([a.get_forces() for a in atoms])
    
    try:
        features = np.load(feature_filename + '.npy')
        feature_gradients = np.load(feature_grad_filename + '.npy')
    except IOError:
        features = featureCalculator.get_featureMat(atoms, show_progress=True)
        np.save(feature_filename, features)
        feature_gradients = featureCalculator.get_all_featureGradients(atoms, show_progress=True)
        np.save(feature_grad_filename, feature_gradients)

    # Apply eta
    Nbins1 = featureCalculator.Nbins1
    Nbondtypes_2body = len(featureCalculator.bondtypes_2body)
    if featureCalculator.use_angular:
        features[:, Nbondtypes_2body*Nbins1:] *= eta
        feature_gradients[:, :, Nbondtypes_2body*Nbins1:] *= 50

    # Permute data
    permut = np.random.permutation(Ndata)
    E = E[permut]
    F = F[permut]
    features = features[permut]
    feature_gradients = feature_gradients[permut]
        
    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    E_predict = np.zeros(Ndata)
    F_predict = np.zeros((Ndata, Natoms*dim))

    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    Ntest = int(np.ceil(Ndata/Nsplit))
    for i in range(Nsplit):
        # Split into test and training
        [i_train1, i_test, i_train2] = np.split(np.arange(Ndata), [i*Ntest, min((i+1)*Ntest, Ndata)])
        i_train = np.r_[i_train1, i_train2]

        # Training data
        features_train = features[i_train]
        E_train = E[i_train]

        # Test data
        features_test = features[i_test]
        feature_gradients_test = feature_gradients[i_test]

        # Perform training
        MAE, params = krr.train(data_values=E_train, featureMat=features_train, add_new_data=False, k=5, **GSkwargs)
        print('MAE:', MAE)
        
        # Perform testing
        E_predict[i_test] = np.array([krr.predict_energy(fnew=f) for f in features_test])
        F_predict[i_test] = np.array([krr.predict_force(fnew=features_test[i], fgrad=feature_gradients_test[i])
                                      for i in range(len(i_test))])

    return E, F, E_predict, F_predict


def correlationPlot(values1, values2, title='', xlabel='', ylabel='', color_weights=None):
    plt.figure()
    max_value = max(np.max(values1), np.max(values2))
    min_value = min(np.min(values1), np.min(values2))
    
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    # Plot line
    line = [min_value,max_value]
    plt.plot(line, line, color='r')
    # Plot points
    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if color_weights is not None:
        cm = plt.cm.get_cmap('RdYlBu_r')
        sc = plt.scatter(values1, values2, c=color_weights, alpha=0.7, vmin=0, vmax=2, cmap=cm)
        cbar = plt.colorbar(sc)
        cbar.set_label('cosine distance')
    else:
        plt.scatter(values1, values2, alpha=0.5)

def distributionPlot(x, title='', xlabel=''):
    plt.figure()
    bins = np.linspace(0, np.max(x), max(int(len(x)/20), 10))
    plt.hist(x, bins=bins)
    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('counts')
    
if __name__ == '__main__':
    atoms = read('graphene_data/graphene_all2.traj', index=':')

    pos_all = np.array([a.get_positions() for a in atoms])
    F_all = np.array([a.get_forces() for a in atoms])

    F0 = F_all[0].reshape(-1)[:5]
    F1 = F_all[1].reshape(-1)[:5]
    
    Fnew = np.r_[F0,F1]
    print(F0.shape)
    print(F1.shape)
    print(Fnew.shape)
    print(F0)
    print(F1)
    print(Fnew)
    
