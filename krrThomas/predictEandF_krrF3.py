import numpy as np
from scipy.spatial.distance import cosine

from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from gaussComparator2 import gaussComparator
from maternComparator import maternComparator
from vector_krr_ase2 import vector_krr_class

import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

import pdb

def predictEandF(atoms_train, atoms_test, featureCalculator, Nsplit=5):
    Ntrain = len(atoms_train)
    Ntest = len(atoms_test)
    
    E_train = np.array([a.get_potential_energy() for a in atoms_train])
    F_train = np.array([a.get_forces() for a in atoms_train])

    E_test = np.array([a.get_potential_energy() for a in atoms_test])
    F_test = np.array([a.get_forces() for a in atoms_test])
    
    features_train = featureCalculator.get_featureMat(atoms_train)
    feature_gradients_train = featureCalculator.get_all_featureGradients(atoms_train)

    features_test = featureCalculator.get_featureMat(atoms_test)
    feature_gradients_test = featureCalculator.get_all_featureGradients(atoms_test)

    # Set up KRR-model
    comparator = maternComparator()
    #comparator = gaussComparator()
    krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

    # Train
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(1,3,10)}
    MAE, params = krr.train(forces=F_train,
                            featureMat=features_train,
                            featureGradMat=feature_gradients_train,
                            add_new_data=False,
                            k=5,
                            **GSkwargs)
    print('MAE:', MAE)
    print(params)
    
    # Perform testing
    E_predict = np.array([krr.predict_energy(fnew=f) for f in features_test]).reshape(-1)
    F_predict = np.array([krr.predict_force(fnew=f, fnew_grad=f_grad)
                          for f, f_grad in zip(features_test, feature_gradients_test)])
    
    return E_test, F_test, E_predict, F_predict


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
    Ndata = len(atoms)
    permutation = np.random.permutation(Ndata)
    atoms = [atoms[i] for i in permutation]
    #atoms = atoms[0::2]
    atoms_train = atoms[:40]
    atoms_test = atoms[40:]
    a0 = atoms[0]
    Ntest = len(atoms_test)
    
    # Setting up the featureCalculator
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    use_angular = False
    gamma = 1
    eta = 20
    
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, use_angular=use_angular)

    # Predicting
    Nsplit = 2

    E, F, Epred, Fpred = predictEandF(atoms_train, atoms_test, featureCalculator, Nsplit=Nsplit)
    print('shape F:', F.shape)
    print('shape Fpred:', Fpred.shape)

    F = F.reshape((Ntest, -1))
    cos_dists = np.array([cosine(Fi, Fpredi) for Fi, Fpredi in zip(F, Fpred)])
    distributionPlot(cos_dists, title='Cosine distance distribution between target and predicted forces\n0=parallel, 1=orthogonal, 2=anti-parallel',
                     xlabel='cosine distance')

    print(F.shape)
    F_norm = np.linalg.norm(F, axis=1)
    Fpred_norm = np.linalg.norm(Fpred, axis=1)
    print(F_norm.shape)
    correlationPlot(Fpred_norm, F_norm, title='Correlation between force magnitude + direction (color)', xlabel='|F| predicted', ylabel='|F| target', color_weights=cos_dists)
    
    # reshape forces
    F = F.reshape(-1)
    Fpred = Fpred.reshape(-1)

    correlationPlot(Epred, E, title='Energy correlation', xlabel='E predicted', ylabel='E target')
    correlationPlot(Fpred, F, title='Correllation between force components', xlabel='F predicted', ylabel='F target')

    plt.show()

    
    
