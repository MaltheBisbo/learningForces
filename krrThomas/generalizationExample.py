import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from delta_functions.delta import delta as deltaFunc
from custom_calculators import krr_calculator

from doubleLJ import doubleLJ_energy_ase as E_doubleLJ

from ase import Atoms
from ase.visualize import view
from ase.data import covalent_radii

def structure(x1,x2):
    '''
    x1, x2 is the two new coordiantes
    d0 is the 
    '''
    d0 = 1
    theta1 = -np.pi/3
    theta2 = -2/3*np.pi
    pos0 = np.array([0,0,0])
    pos1 = np.array([-d0 + x1*np.cos(theta1), x1*np.sin(theta1), 0])
    pos2 = np.array([d0 + x2*np.cos(theta2), x2*np.sin(theta2), 0])

    pos = np.array([pos1, pos0, pos2])

    a = Atoms('3He',
              positions=pos,
              pbc=[0,0,0],
              cell=[3,3,3])
    return a

def structure_list(x1_list, x2_list):
    a_list = [structure(x1, x2) for x1, x2 in zip(x1_list, x2_list)]
    return a_list

class doubleLJ_delta():
    def __init__(self, frac):
        self.frac = frac

    def energy(self, a):
        return self.frac * E_doubleLJ(a)


a0 = structure(0,2)


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
#delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                #delta_function=delta_function,
                bias_fraction=0.7,
                bias_std_add=1)

delta_function = doubleLJ_delta(0.1)
krr_delta01 = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator,
                        delta_function=delta_function,
                        bias_fraction=0.7,
                        bias_std_add=1)

delta_function = doubleLJ_delta(0.2)
krr_delta02 = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator,
                        delta_function=delta_function,
                        bias_fraction=0.7,
                        bias_std_add=1)


# Generate training data
train_coord = np.linspace(-0.2,0.2,3)
x1_train = np.r_[train_coord-0.2]  # , train_coord+2]
x2_train = np.r_[-train_coord+2.2]  # , train_coord]
#x1_train = np.r_[train_coord, train_coord+2]
#x2_train = np.r_[train_coord+2, -train_coord]

a_train = structure_list(x1_train, x2_train)
E_train = [E_doubleLJ(a) for a in a_train]


# Train model
sigma = 10
GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}
#GSkwargs = {'reg': [1e-7], 'sigma': np.logspace(0,2,10)}
MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, **GSkwargs)
print(MAE, params)
MAE, params = krr_delta01.train(atoms_list=a_train, data_values=E_train, k=3, **GSkwargs)
print(MAE, params)
MAE, params = krr_delta02.train(atoms_list=a_train, data_values=E_train, k=3, **GSkwargs)
print(MAE, params)



# Generate test data
Npoints = 100
test_coord = np.linspace(-0.5, 2.5, Npoints)
x1_test = np.linspace(-0.5, 2.5, Npoints)
x2_test = test_coord

# Make grid of test data for contour plot
X1, X2 = np.meshgrid(x1_test, x2_test)
X1_flat = X1.reshape(-1)
X2_flat = X2.reshape(-1)
a_grid = structure_list(X1_flat, X2_flat)

E_grid = np.array([krr.predict_energy(a) for a in a_grid]).reshape((Npoints,Npoints))
E_grid_true = np.array([E_doubleLJ(a) for a in a_grid]).reshape((Npoints,Npoints))
E_grid_true[E_grid_true > 1] = 0

x1_path = test_coord
x2_path = test_coord[::-1]
a_path = structure_list(x1_path, x2_path)
Epred_path = np.array([krr.predict_energy(a) for a in a_path])
Etrue_path = np.array([E_doubleLJ(a) for a in a_path])

Epred_path_delta01 = np.array([krr_delta01.predict_energy(a) for a in a_path])
Epred_path_delta02 = np.array([krr_delta02.predict_energy(a) for a in a_path])

plt.figure()
plt.title('Predicted energy landscape \nsigma={}, no delta'.format(sigma))
plt.xlabel('x1')
plt.ylabel('x2')
plt.contourf(X1, X2, E_grid)
plt.plot([-0.5,2.5], [2.5,-0.5], 'r:')
plt.plot(x1_train, x2_train, 'kx')
plt.colorbar()

plt.figure()
plt.title('Target energy landscape')
plt.xlabel('x1')
plt.ylabel('x2')
plt.contourf(X1, X2, E_grid_true)
plt.colorbar()

plt.figure()
plt.title('Energy of linear path \nsigma={} for ML models'.format(sigma))
plt.xlabel('x1')
plt.ylabel('Energy')
plt.plot(test_coord, Etrue_path, label='Target')
plt.plot(test_coord, Epred_path, 'k:', label='ML no delta')
plt.plot(test_coord, Epred_path_delta01, 'r:', label='ML delta=0.1*dLJ')
plt.plot(test_coord, Epred_path_delta02, 'g:', label='ML delta=0.2*dLJ')
plt.legend()
plt.show()

