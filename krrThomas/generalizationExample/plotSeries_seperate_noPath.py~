import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
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
comparator = gaussComparator()
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                bias_fraction=0.7,
                bias_std_add=1)
sigma = 20
GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}

# // TRAINING DATA //
"""
# base points
train_coord = np.linspace(-0.2,0.2,3)
x1_train = np.r_[train_coord-0.0, ]
x2_train = np.r_[-train_coord+2.0]

# new points
x1_train = np.r_[x1_train, 0.745, 0.85]
x2_train = np.r_[x2_train, 0.745, 0.85]

a_train = structure_list(x1_train, x2_train)
E_train = [E_doubleLJ(a) for a in a_train]
""" 
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

# // PATH //

# Corner path
#path_coord1 = np.linspace(-0.5, 1, Npoints)
#path_coord2 = np.linspace(1, 2.5, Npoints)
#x1_path = np.r_[path_coord1, path_coord1[::-1]]
#x2_path = np.r_[path_coord2[::-1], path_coord1[::-1]]

path_coord1 = np.linspace(-0.5, 1, Npoints)
path_coord2 = np.linspace(1, 2.5, Npoints)
x1_path = np.r_[path_coord1, path_coord2]
x2_path = np.r_[path_coord1, path_coord1[::-1]]


# Straight path
#x1_path = test_coord
#x2_path = test_coord[::-1]

a_path = structure_list(x1_path, x2_path)


fontsize1 = 21
fontsize2 = 19
Ntrain = len(a_train)

for i in range(1,Ntrain+1):
    x1_train_sub = x1_train[:i]
    x2_train_sub = x2_train[:i]
    a_train_sub = a_train[:i]
    E_train_sub = E_train[:i]
    
    # Train model
    MAE, params = krr.train(atoms_list=a_train_sub, data_values=E_train_sub, k=i, **GSkwargs)
    print(MAE, params)

    E_grid = np.array([krr.predict_energy(a) for a in a_grid]).reshape((Npoints,Npoints))

    # path energies
    Epred_path = []
    Epred_path_error = []
    for a in a_path:
        E, error, _ = krr.predict_energy(a, return_error=True)
        Epred_path.append(E)
        Epred_path_error.append(error)
    Epred_path = np.array(Epred_path)
    Epred_path_error = np.array(Epred_path_error)
    #Epred_path = np.array([krr.predict_energy(a) for a in a_path])
    Etrue_path = np.array([E_doubleLJ(a) for a in a_path])

    v = np.linspace(-9.0, 2.0, 12, endpoint=True)

    
    ax1 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    #ax1.tick_params(labelsize=fontsize2)
    plt.xticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    
    plt.title('True energy landscape', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('x2', fontsize=fontsize1)
    plt.contourf(X1, X2, E_grid_true, v)
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
    plt.savefig('results/3body/seperate/TrueContour{}.pdf'.format(i), transparent=True)

    
    ax2 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(np.arange(0, 3, step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0, 3, step=1), fontsize=fontsize2)

    #plt.title('Model energy landscape \nsigma={}, no delta'.format(sigma))
    plt.title('Model energy landscape', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('x2', fontsize=fontsize1)
    plt.contourf(X1, X2, E_grid, v)
    plt.plot(x1_path, x2_path, 'r:')
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
    plt.savefig('results/3body/seperate/PredictedContour{}.pdf'.format(i), transparent=True)

    
    ax3 = plt.figure()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.xticks(np.arange(0,3,step=1), fontsize=fontsize2)
    plt.yticks(np.arange(-8,1,step=2), fontsize=fontsize2)
    plt.title('Energy of path', fontsize=fontsize1)
    plt.xlabel('x1', fontsize=fontsize1)
    plt.ylabel('Energy', fontsize=fontsize1)
    plt.xlim([-0.5, 2.5])
    plt.ylim([-8.2,0.2])
    
    plt.plot(x1_path, Etrue_path, 'k', label='True', lw=2.5)
    plt.plot(x1_path, Epred_path, label='Model', lw=2)
    plt.fill_between(x1_path, Epred_path+2*Epred_path_error, Epred_path-2*Epred_path_error, facecolor='blue', alpha=0.3)

    # Plot seperating line
    ylim_min, ylim_max = plt.ylim()
    plt.plot([1,1], [ylim_min, ylim_max], 'k')
    #plt.ylim([ylim_min, ylim_max])

    # plot training points
    if i == 6:
        plt.plot(x1_train_sub[:-1], E_train_sub[:-1], color='r', marker='o', linestyle='None', label='Data')
        plt.plot(x1_train_sub[-1], E_train_sub[-1], color='g', marker='o', linestyle='None', label='GM')
        for n,[x1,E] in enumerate(zip(x1_train_sub[:-1], E_train_sub[:-1])):
            plt.text(x1-0.03, E-0.7, '{}'.format(n+1), fontsize=fontsize2)
    else:
        plt.plot(x1_train_sub, E_train_sub, color='r', marker='o', linestyle='None', label='Data')
        for n,[x1,E] in enumerate(zip(x1_train_sub, E_train_sub)):
            plt.text(x1-0.03, E-0.7, '{}'.format(n+1), fontsize=fontsize2)
    plt.legend(loc=4, fontsize=fontsize2 - 2)
    plt.savefig('results/3body/seperate/path{}.pdf'.format(i), transparent=True)
    """
    # Plot bias
    xlim_min, xlim_max = plt.xlim()
    plt.plot([xlim_min, xlim_max], [0,0], 'k:', label='bias')
    plt.xlim([xlim_min, xlim_max])
    """


    
    features = np.array([featureCalculator.get_feature(a) for a in a_train_sub])

    ax4 = plt.figure(figsize=(6.7,4.6))
    plt.subplots_adjust(bottom=0.15)
    #ax4.tick_params(labelsize=fontsize2)
    plt.xticks(np.arange(0,4,step=1), fontsize=fontsize2)
    plt.yticks(np.arange(0,3,step=1), fontsize=fontsize2)
    plt.title('Descriptors', fontsize=fontsize1)
    plt.ylim([0,2.6])
    plt.xlabel('r', fontsize=fontsize1)
    plt.ylabel('F', fontsize=fontsize1)
    for n, a in enumerate(a_train_sub):
        feature = featureCalculator.get_feature(a)
        if n == 5:
            plt.plot(binwidth1*np.arange(len(feature)), feature, label='GM', color='g', lw=3.5)
        else:
            plt.plot(binwidth1*np.arange(len(feature)), feature, label='Data {}'.format(n+1), lw=2)
        
    plt.legend(fontsize=fontsize2 - 2)
    plt.savefig('results/3body/seperate/features{}.pdf'.format(i), transparent=True)
    

a_coordinateExample = structure(0,0)
plt.figure(figsize=(1,1))
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plotStruct(a_coordinateExample)
plt.xlim([-1.7, 1.7])
plt.ylim([-2.5, 0.7])
plt.savefig('results/3body/seperate/coordExample_structure.pdf', transparent=True)
    
    
for i, a in enumerate(a_train):
    if i == 0:
        print(a.get_positions())
    plt.figure(figsize=(1,1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    if i == 5:
        plotStruct(a, color='g')
    else:
        plotStruct(a)

    plt.xlim([-1.7, 1.7])
    plt.ylim([-2.5, 0.7])
    plt.savefig('results/3body/seperate/TrainingStructures_sep{}.pdf'.format(i), transparent=True)

