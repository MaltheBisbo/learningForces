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

from helpFunctions import cartesian_coord, structure, structure_list

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
sigma = 10
GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}

# // TRAINING DATA //

# base points
train_coord = np.linspace(-0.2,0.2,3)
x1_train = np.r_[train_coord-0.0, ]
x2_train = np.r_[-train_coord+2.0]

# new points
x1_train = np.r_[x1_train, 0.745, 0.85]
x2_train = np.r_[x2_train, 0.745, 0.85]

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
E_grid_true[E_grid_true > 1] = 0

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
    plt.figure(figsize=(15,13))
    plt.subplots_adjust(left=1/14, right=1-1/14, wspace=2/14,
                        bottom=1/14, top=1-1/14, hspace=4/14)
    plt.subplot(2,2,1)
    
    plt.title('Target energy landscape')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid_true, v)
    plt.colorbar()
    
    
    plt.subplot(2,2,2)
    plt.title('Predicted energy landscape \nsigma={}, no delta'.format(sigma))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid, v)
    plt.plot(x1_path, x2_path, 'r:')
    plt.plot(x1_train_sub, x2_train_sub, color='r', marker='o', linestyle='None')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('Energy of linear path \nsigma={} for ML models'.format(sigma))
    plt.xlabel('x2')
    plt.ylabel('Energy')
    plt.plot(x2_path, Etrue_path, 'k-.', label='Target')
    plt.plot(x2_path, Epred_path, label='ML no delta')
    plt.fill_between(x2_path, Epred_path+Epred_path_error, Epred_path-Epred_path_error, facecolor='blue', alpha=0.3)
    
    # plot training points
    plt.plot(x2_train_sub, E_train_sub, color='r', marker='o', linestyle='None')
    
    # Plot bias
    xlim_min, xlim_max = plt.xlim()
    plt.plot([xlim_min, xlim_max], [0,0], 'k:', label='bias')
    plt.xlim([xlim_min, xlim_max])

    # Plot seperating line
    ylim_min, ylim_max = plt.ylim()
    plt.plot([1,1], [ylim_min, ylim_max], 'k')
    plt.ylim([ylim_min, ylim_max])
    plt.legend()

    features = np.array([featureCalculator.get_feature(a) for a in a_train_sub])
    plt.subplot(2,2,4)
    plt.title('Features')
    plt.xlabel('Interatomic distance')
    plt.ylabel('Feature magnitude')
    for n, a in enumerate(a_train_sub):
        feature = featureCalculator.get_feature(a)
        plt.plot(binwidth1*np.arange(len(feature)), feature, label='struktur {}'.format(n))
    plt.legend()
    
    
    #plt.savefig('results/3body/Ntrain{0:d}_sig{1:d}_eta{2:d}.pdf'.format(i, sigma, eta))
    plt.savefig('results/3body/NtrainNoAng{0:d}_sig{1:d}_eta{2:d}.pdf'.format(i, sigma, eta))


def plotStruct(a, x0, y0):
    pos = a.get_positions()[:,:2]
    pos += np.array([x0,y0])
    plt.plot(pos[:,0], pos[:,1], 'ro', ms=20)
    
    
        
        
plt.figure()
for i, a in enumerate(a_train):
    plotStruct(a_train[0], 0,-3*i)

plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()
