import numpy as np
from scipy.spatial.distance import cosine

from angular_fingerprintFeature_test import Angular_Fingerprint
from gaussComparator import gaussComparator
from krr_ase import krr_class

import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb

def predictE(atoms_train, atoms_predict, featureCalculator, feature_filename, eta=1):

    Etrain = np.array([a.get_potential_energy() for a in atoms_train])
    
    try:
        features = np.load(feature_filename + '.npy')
    except IOError:
        features = featureCalculator.get_featureMat(atoms_train, show_progress=True)
        np.save(feature_filename, features)

    # Apply eta
    Nbins1 = featureCalculator.Nbins1
    Nbondtypes_2body = len(featureCalculator.bondtypes_2body)
    if featureCalculator.use_angular:
        features[:, Nbondtypes_2body*Nbins1:] *= eta

    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    # Perform training
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
    FVU, params = krr.train(data_values=Etrain, featureMat=features, add_new_data=False, k=5, **GSkwargs)

    # Predict
    Npredict = len(atoms_predict)
    Epred = np.zeros(Npredict)
    Epred_error = np.zeros(Npredict)
    theta0 = np.zeros(Npredict)
    for i, a in enumerate(atoms_predict):
        Epred[i], Epred_error[i], theta0[i] = krr.predict_energy(atoms=a, return_error=True)

    return Epred, Epred_error, theta0

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
    atoms_train = read('graphene_data/graphene_all2.traj', index=':')
    atoms_predict = read('grapheneMLrelax/grapheneAng_train-0.traj', index=':')
    Ntrain = len(atoms_train)
    Npredict = len(atoms_predict)
    a0 = atoms_train[0]

    Epred_relax = np.array([a.get_potential_energy() for a in atoms_predict])
    # Setting up the featureCalculator
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    use_angular = True
    gamma = 1
    eta = 20
    
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)

    # Predicting
    filepath = 'grapheneMLrelax/correlation/features/'
    feature_filename = filepath + 'features_all2'

    Epred, Epred_error, theta0 = predictE(atoms_train, atoms_predict, featureCalculator, feature_filename=feature_filename, eta=eta)

    plt.figure()
    plt.plot(np.arange(Npredict), Epred, color='blue')
    plt.plot(np.arange(Npredict), Epred_relax, color='red')

    plt.figure()
    plt.plot(np.arange(Npredict), Epred_error)
    plt.plot(np.arange(Npredict), np.sqrt(theta0))
    plt.show()

    
    
