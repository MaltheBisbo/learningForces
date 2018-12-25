import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from GPR import GPR as krr_class
#from delta_functions.delta import delta as deltaFunc
from custom_calculators import krr_calculator

from doubleLJ import doubleLJ_energy_ase as E_doubleLJ

from ase import Atoms
from ase.visualize import view
from ase.data import covalent_radii

from helpFunctions import *

a0 = structure(0,2)

# Set up featureCalculator
Rc1 = 3
binwidth1 = 0.025
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 5
use_angular = False

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)


# Set up KRR-model
sigma1 = 20
kwargs = {'sigma': sigma1}
comparator = gaussComparator(**kwargs)
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                regGS=[1e-7])

sigma2 = 20
kwargs2 = {'sigma': sigma2}
comparator2 = gaussComparator(**kwargs2)
krr2 = krr_class(comparator=comparator2,
                 featureCalculator=featureCalculator,
                 regGS=[1e-7])

# // TRAINING DATA //

coord_train = np.array([[0.0, 2.0],
                        [-0.2, 2.2],
                        [0.2, 1.8],
                        [0.25, 0.25],
                        [-0.1, -0.1],
                        [1,1]])  # Last one is the global minimum
coord_train = np.array([[2.0, 0.0],
                        [2.2, -0.2],
                        [1.8, 0.2],
                        [0.20, 0.20],
                        [-0.1, -0.1],
                        [1,1]])  # Last one is the global minimum
x1_train = coord_train[:,0]
x2_train = coord_train[:,1]

a_train = structure_list(x1_train, x2_train)
E_train = [E_doubleLJ(a) for a in a_train]

# // TEST DATA //

Npoints = 100
test_coord = np.linspace(-0.5, 2.5, Npoints)
x1_test = np.linspace(-0.5, 2.5, Npoints)
x2_test = test_coord

# Make grid of test data for contour plot
X1, X2 = np.meshgrid(x1_test, x2_test)
X1_flat = X1.reshape(-1)
X2_flat = X2.reshape(-1)
a_grid = structure_list(X1_flat, X2_flat)

E_grid_true = np.array([E_doubleLJ(a) for a in a_grid]).reshape((Npoints,Npoints))
E_grid_true[E_grid_true > 2] = 2

fontsize1 = 21
fontsize2 = 19
Ntrain = len(a_train)

for i in range(1,Ntrain+1):
    x1_train_sub = x1_train[:i]
    x2_train_sub = x2_train[:i]
    a_train_sub = a_train[:i]
    E_train_sub = np.array(E_train[:i])
    
    # Train model
    MAE, params = krr.train(atoms_list=a_train_sub, data_values=E_train_sub, k=i, add_new_data=False)
    print(MAE, params)

    E_grid = np.array([krr.predict_energy(a) for a in a_grid]).reshape((Npoints,Npoints))

    # Train meta
    MAE, params = krr2.train(atoms_list=a_train_sub, data_values=E_train_sub, k=i, add_new_data=False)
    
    krr2.alpha = np.ones(len(krr2.alpha))
    meta_grid = np.array([krr2.predict_energy(a) for a in a_grid]).reshape((Npoints,Npoints))
    print(krr2.alpha)
    
    v1 = np.linspace(-9.0, 2.0, 12, endpoint=True)
    v2 = np.linspace(0, 6.0, 13, endpoint=True)

    
    ax1 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    #ax1.tick_params(labelsize=fontsize2)
    plt.xticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    
    plt.title('True energy landscape', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('x2', fontsize=fontsize1)
    plt.contourf(X1, X2, E_grid_true, v1)
    if i == 6:
        plt.plot(x1_train_sub[:-1], x2_train_sub[:-1], color='r', marker='o', linestyle='None')
        plt.plot(x1_train_sub[-1], x2_train_sub[-1], color='g', marker='o', linestyle='None')
    else:
        plt.plot(x1_train_sub, x2_train_sub, color='r', marker='o', linestyle='None')
    for n,[x1,x2] in enumerate(zip(x1_train_sub[:], x2_train_sub[:])):
        if n < 5:
            plt.text(x1-0.05, x2+0.1, '{}'.format(n+1), fontsize=fontsize2)
    cb1 = plt.colorbar()
    cb1.ax.tick_params(labelsize=fontsize2)
    plt.savefig('results/metaDyn/True{}.pdf'.format(i), transparent=True)

    
    ax2 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0, 3, step=1), fontsize=fontsize2)

    #plt.title('Model energy landscape \nsigma={}, no delta'.format(sigma))
    plt.title('Model energy landscape', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('x2', fontsize=fontsize1)
    plt.contourf(X1, X2, E_grid, v1)
    if i == 6:
        plt.plot(x1_train_sub[:-1], x2_train_sub[:-1], color='r', marker='o', linestyle='None')
        plt.plot(x1_train_sub[-1], x2_train_sub[-1], color='g', marker='o', linestyle='None')
    else:
        plt.plot(x1_train_sub, x2_train_sub, color='r', marker='o', linestyle='None')
    for n,[x1,x2] in enumerate(zip(x1_train_sub[:], x2_train_sub[:])):
        if n < 5:
            plt.text(x1-0.05, x2+0.1, '{}'.format(n+1), fontsize=fontsize2)
    cb2 = plt.colorbar()
    cb2.ax.tick_params(labelsize=fontsize2)
    plt.savefig('results/metaDyn/Pred{}_sig{}.pdf'.format(i, sigma1), transparent=True)

    
    ax3 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0, 3, step=1), fontsize=fontsize2)

    #plt.title('Model energy landscape \nsigma={}, no delta'.format(sigma))
    plt.title('Model energy landscape', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('x2', fontsize=fontsize1)
    plt.contourf(X1, X2, meta_grid, v2)
    if i == 6:
        plt.plot(x1_train_sub[:-1], x2_train_sub[:-1], color='r', marker='o', linestyle='None')
        plt.plot(x1_train_sub[-1], x2_train_sub[-1], color='g', marker='o', linestyle='None')
    else:
        plt.plot(x1_train_sub, x2_train_sub, color='r', marker='o', linestyle='None')
    for n,[x1,x2] in enumerate(zip(x1_train_sub[:], x2_train_sub[:])):
        if n < 5:
            plt.text(x1-0.05, x2+0.1, '{}'.format(n+1), fontsize=fontsize2)
    cb2 = plt.colorbar()
    cb2.ax.tick_params(labelsize=fontsize2)
    plt.savefig('results/metaDyn/sig{}_meta{}.pdf'.format(sigma2, i), transparent=True)
    
