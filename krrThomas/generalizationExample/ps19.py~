import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from delta_functions.delta import delta as deltaFunc
from custom_calculators import krr_calculator

from doubleLJ import doubleLJ_energy_ase as E_doubleLJ

from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.data import covalent_radii

from helpFunctionsDLJ19 import get_structure, get_structure_list


# // TRAINING DATA //

# base points
#a_train_relaxed = read('dLJ19data/relaxed.traj', index=':')
a_train_unrelaxed = read('dLJ19data/unrelaxed.traj', index=':')
#a_train = a_train_relaxed
a_train = a_train_unrelaxed
Ntrain = len(a_train)
#permut = np.random.permutation(Ntrain)
#a_train = [a_train[i] for i in permut]
E_train = [E_doubleLJ(a) for a in a_train]

a0 = a_train[0]

# Set up featureCalculator
Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 30
use_angular = True

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)


# Set up KRR-model
comparator = gaussComparator()
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                bias_fraction=0.7,
                bias_std_add=1)




# // TEST DATA //

Npoints = 50
x1_test = np.linspace(-0.5, 1.5, Npoints)
x2_test = np.linspace(-0.5, 1.5, Npoints)

# Make grid of test data for contour plot
X1, X2 = np.meshgrid(x1_test, x2_test)
X1_flat = X1.reshape(-1)
X2_flat = X2.reshape(-1)
a_grid = get_structure_list(X1_flat, X2_flat, a0)

E_grid_true = np.array([E_doubleLJ(a) for a in a_grid]).reshape((Npoints,Npoints))
E_grid_true[E_grid_true > 1] = 0

# // PATH //

# Corner path
x1_path = np.linspace(0, 1, Npoints)
x2_path = np.linspace(0, 1, Npoints)

a_path = get_structure_list(x1_path, x2_path, a0)


Nseries = 5
Ntrain_list = np.logspace(np.log10(5), np.log10(Ntrain), Nseries).astype(int)
sigma = 10
GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}
for i in range(Nseries):
    Ntrain = Ntrain_list[i]
    # x1_train_sub = x1_train[:i]
    # x2_train_sub = x2_train[:i]
    a_train_sub = a_train[:Ntrain]
    E_train_sub = E_train[:Ntrain]
    
    # Train model
    MAE, params = krr.train(atoms_list=a_train_sub, data_values=E_train_sub, k=5, **GSkwargs)
    bias = krr.beta
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
    
    plt.figure(figsize=(15,13))
    plt.subplots_adjust(left=1/14, right=1-1/14, wspace=2/14,
                        bottom=1/14, top=1-1/14, hspace=4/14)
    plt.subplot(2,2,1)
    
    plt.title('Target energy landscape')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid_true)
    plt.colorbar()
    
    
    plt.subplot(2,2,2)
    plt.title('Predicted energy landscape \nsigma={}, no delta'.format(sigma))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid)
    plt.plot(x1_path, x2_path, 'r:')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('Energy of linear path \nsigma={} for ML models, bias=mean(Etrain)={}'.format(sigma, bias))
    plt.xlabel('x2')
    plt.ylabel('Energy')
    plt.plot(x2_path, Etrue_path, 'k-.', label='Target')
    plt.plot(x2_path, Epred_path, 'b-', label='ML no delta')
    plt.fill_between(x2_path, Epred_path+Epred_path_error, Epred_path-Epred_path_error, facecolor='blue', alpha=0.3)

    # Set ylim
    plt.ylim([np.min(Etrue_path)-2, np.max(Etrue_path) + 2])
    
    # Plot bias
    xlim_min, xlim_max = plt.xlim()
    plt.plot([xlim_min, xlim_max], [bias,bias], 'k:', label='bias')
    plt.xlim([xlim_min, xlim_max])

    plt.savefig('results/19body/Ntrain{}_unrel.pdf'.format(Ntrain))
    
