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
binwidth1 = 0.05
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
coord_train = np.array([[-0.2, 2.2],
                        [0.2, 1.8],
                        [0.0, 2.0],
                        [0.25,0.25],
                        [-0.1,-0.1]])
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
path_coord1 = np.linspace(-0.5, 1, Npoints)
path_coord2 = np.linspace(1, 2.5, Npoints)
x1_path = np.r_[path_coord1, path_coord1[::-1]]
x2_path = np.r_[path_coord2[::-1], path_coord1[::-1]]

# Straight path
#x1_path = test_coord
#x2_path = test_coord[::-1]

a_path = structure_list(x1_path, x2_path)



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

    plt.figure(figsize=(12,11))
    plt.subplots_adjust(left=1/12, right=1-1/12, wspace=2/12,
                        bottom=1/11, top=1-1/11, hspace=4/11)
    #plt.figure(figsize=(15,13))
    #plt.subplots_adjust(left=1/14, right=1-1/14, wspace=2/14,
    #                    bottom=1/14, top=1-1/14, hspace=4/14)
    plt.subplot(2,2,1)
    
    plt.title('True energy landscape')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid_true, v)
    plt.colorbar()
    
    
    plt.subplot(2,2,2)
    #plt.title('Model energy landscape \nsigma={}, no delta'.format(sigma))
    plt.title('Model energy landscape')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid, v)
    plt.plot(x1_path, x2_path, 'r:')
    plt.plot(x1_train_sub, x2_train_sub, color='r', marker='o', linestyle='None')
    for n,[x1,x2] in enumerate(zip(x1_train_sub, x2_train_sub)):
        plt.text(x1-0.05, x2+0.1, '{}'.format(n+1), fontsize=13)
    plt.colorbar()

    plt.subplot(2,2,3)
    #plt.title('Energy of linear path \nsigma={} for ML models'.format(sigma))
    plt.title('Energy of linear path')
    plt.xlabel('x2')
    plt.ylabel('Energy')
    plt.plot(x2_path, Etrue_path, 'k-.', label='Target')
    plt.plot(x2_path, Epred_path, label='ML no delta')
    plt.fill_between(x2_path, Epred_path+2*Epred_path_error, Epred_path-2*Epred_path_error, facecolor='blue', alpha=0.3)
    
    # plot training points
    plt.plot(x2_train_sub, E_train_sub, color='r', marker='o', linestyle='None')

    plt.xlim([-0.5, 2.5])
    plt.ylim([-8,0.2])
    """
    # Plot bias
    xlim_min, xlim_max = plt.xlim()
    plt.plot([xlim_min, xlim_max], [0,0], 'k:', label='bias')
    plt.xlim([xlim_min, xlim_max])
    """

    # Plot seperating line
    ylim_min, ylim_max = plt.ylim()
    plt.plot([1,1], [ylim_min, ylim_max], 'k')
    plt.ylim([ylim_min, ylim_max])
    plt.legend(loc=4)

    features = np.array([featureCalculator.get_feature(a) for a in a_train_sub])
    plt.subplot(2,2,4)
    plt.title('Features')
    plt.ylim([0,2.2])
    plt.xlabel('Interatomic distance')
    plt.ylabel('Feature magnitude')
    for n, a in enumerate(a_train_sub):
        feature = featureCalculator.get_feature(a)
        plt.plot(binwidth1*np.arange(len(feature)), feature, label='Structure {}'.format(n+1))
    plt.legend()
    
    
    #plt.savefig('results/3body/Ntrain{0:d}_sig{1:d}_eta{2:d}.pdf'.format(i, sigma, eta))
    plt.savefig('results/3body/NtrainNoAng{0:d}_sig{1:d}_eta{2:d}.pdf'.format(i, sigma, eta))


    
    dx = 4
    dy = 3
    Nstruct = len(a_train_sub)
    plt.figure(figsize=(4.5,11))
    plt.subplots_adjust(left=1/12, right=1-1/12,
                        bottom=1/11, top=1-1/11)
    #plt.xlim([-(dx+1.1)/2, (dx+1.1)/2])
    plt.ylim([-(5+0.1)*dy, 1.9*dy])
    # Plot coordinate example
    plt.text(-0.72*dx, 1.93*dy, 'Coordinate definition', fontsize=15)
    plotCoordinateExample(0,1.5*dy, scale=1)
    plt.plot([-0.7*dx, -0.7*dx, dx/2, dx/2, -0.7*dx], [1.8*dy, 0.8*dy, 0.8*dy, 1.8*dy, 1.8*dy], 'k', lw=3)
    
    
    # Plot training data
    boxcolor = 'steelblue'
    plt.text(-0.70*dx, 0.38*dy, 'Training structures', fontsize=15)
    plt.plot([-0.7*dx, dx/2], [1/4*dy, 1/4*dy], boxcolor, lw=2)
    for i, a in enumerate(a_train_sub):
        plt.text(-0.6*dx, -dy*i, '{})'.format(i+1), fontsize=15)
        plotStruct(a_train[i], 0,-dy*i)
        plt.plot([-0.7*dx, dx/2], [-(3/4+i)*dy, -(3/4+i)*dy], boxcolor, lw=2)
    plt.plot([-0.7*dx, -0.7*dx], [1/4*dy, (1/4-Nstruct)*dy], boxcolor, lw=2)
    plt.plot([dx/2, dx/2], [1/4*dy, (1/4-Nstruct)*dy], boxcolor, lw=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('results/3body/TrainingStructures{}.pdf'.format(i))


