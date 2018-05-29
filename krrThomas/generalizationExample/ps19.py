import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from delta_functions.delta import delta as deltaFunc
from custom_calculators import krr_calculator

from doubleLJ import doubleLJ_energy_ase as E_doubleLJ


from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.data import covalent_radii

#from helpFunctionsDLJ19 import get_structure, get_structure_list


def get_structure(c1, c2, a0, d0=1.06):
    d0 = 1.06
    dy = d0*np.sin(np.pi/3)
    dx = d0*np.cos(np.pi/3)
    
    row1 = np.c_[d0*np.arange(3).T, np.zeros(3).T, np.zeros(3).T]
    row2 = np.c_[d0*np.arange(4).T-dx, dy*np.ones(4).T, np.zeros(4).T]
    row3 = np.c_[d0*np.arange(5).T-2*dx, 2*dy*np.ones(5).T, np.zeros(5).T]
    row4 = np.c_[d0*np.arange(4).T-dx, 3*dy*np.ones(4).T, np.zeros(4).T]
    row5 = np.c_[d0*np.arange(3).T, 4*dy*np.ones(3).T, np.zeros(3).T]

    row5[:,0] -= c1*d0

    row3[4,0] += c2*dx
    row3[4,1] += c2*dy

    row2[3,0] += c2*dx
    row2[3,1] += c2*dy
    
    positions = np.r_[row1, row2, row3, row4, row5]
    structure = a0.copy()
    structure.set_positions(positions)
    #structure = Atoms('19He', positions=positions)
    return structure

def plotStruct(a):
    pos = a.get_positions()
    x = pos[:,0]
    y = pos[:,1]
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    k = 0.4
    plt.figure(figsize=(k*dx,k*(dy+1)))
    plt.scatter(x, y, c='r', marker='o', edgecolors='k', s=60)
    plt.gca().set_aspect('equal', adjustable='box')

def get_distance(a1, a2, featureCalculator):
    f1 = featureCalculator.get_feature(a1)
    f2 = featureCalculator.get_feature(a2)
    d = euclidean(f1, f2)
    return d

def get_distance_center(f_center, a, featureCalculator):
    f = featureCalculator.get_feature(a)
    d = euclidean(f_center, f)
    return d

def get_center_feature(a_list, featureCalculator):
    fMat = np.array([featureCalculator.get_feature(a) for a in a_list])
    f_mean = fMat.mean(axis=0)
    return f_mean
    
a_base = read('dLJ19data/relaxed.traj', index='0')
a0 = get_structure(0,0,a_base)
a1 = get_structure(1,0,a_base)
a2 = get_structure(0,1,a_base)
a3 = get_structure(1,1,a_base)
a_main_list = [a0,a1,a2,a3]

plotStruct(a0)
plotStruct(a1)
plotStruct(a2)
plotStruct(a3)


# // TRAINING DATA //

"""
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
"""

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

f_center = get_center_feature([a0,a1,a2,a3], featureCalculator)
d_list = [get_distance_center(f_center, a, featureCalculator) for a in [a0,a1,a2,a3]]
print(d_list)

print(get_distance(a1,a2,featureCalculator))

"""
a_train = []
for i_relax in range(200):
    label = '../doubleLJproject/LJdata/LJ19/relax{}'.format(i_relax)
    traj = read(label + '.traj', index=':-2')
    Nsteps = len(traj)
    index_random = np.random.permutation(Nsteps)
    for n in range(Nsteps):
        print(n)
        index_sample = index_random[n]
        a_new = traj[index_sample]
        d_array = np.array([get_distance(a_new, a, featureCalculator) for a in a_main_list])
        if np.max(d_array) > 5:
            a_train.append(a_new)
            break
write('data_N200_d5.traj', a_train)
"""

d = 5
a_train = read('data_N200_d5.traj', index=':')
E_train = np.array([E_doubleLJ(a) for a in a_train])
plotStruct(a_train[0])
plotStruct(a_train[1])
plotStruct(a_train[2])
plotStruct(a_train[3])
plotStruct(a_train[4])
#plt.show()

# Set up KRR-model
comparator = gaussComparator()
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator)
sigma = 10
#GSkwargs = {'reg': [1e-7], 'sigma': [sigma]}
GSkwargs = {'reg': [1e-7], 'sigma': np.logspace(0,3,10)}


# // TEST DATA //

Npoints = 30
x1_test = np.linspace(-0.1, 1.1, Npoints)
x2_test = np.linspace(-0.1, 1.1, Npoints)

# Make grid of test data for contour plot
X1, X2 = np.meshgrid(x1_test, x2_test)
X1_flat = X1.reshape(-1)
X2_flat = X2.reshape(-1)
a_grid = [get_structure(x1, x2, a_base) for x1, x2 in zip(X1_flat, X2_flat)]
#a_grid = get_structure_list(X1_flat, X2_flat, a_base)

E_grid_true = np.array([E_doubleLJ(a) for a in a_grid]).reshape((Npoints,Npoints))
E_grid_true[E_grid_true > 1] = 0






"""
# // PATH //

# Corner path
x1_path = np.linspace(0, 1, Npoints)
x2_path = np.linspace(0, 1, Npoints)

a_path = get_structure_list(x1_path, x2_path, a0)
"""

Ntrain_max = 100
Nseries = 5
Ntrain_list = np.logspace(np.log10(5), np.log10(Ntrain_max), Nseries).astype(int)
#Ntrain_list = [20, 100]
#Nseries = len(Ntrain_list)

v = np.linspace(-115, -92.5, 10, endpoint=True)

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

    """
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
    """

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
    plt.title('Predicted energy landscape \nNtrain={}'.format(sigma, Ntrain))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.contourf(X1, X2, E_grid, v)
    #plt.plot(x1_path, x2_path, 'r:')
    plt.colorbar()

    """
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
    """
    
    plt.savefig('results/19body/Ntrain{}_d{}.pdf'.format(Ntrain, d))

