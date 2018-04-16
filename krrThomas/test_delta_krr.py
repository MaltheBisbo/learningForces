import numpy as np

from gaussComparator import gaussComparator
#from angular_fingerprintFeature import Angular_Fingerprint
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from delta_functions.delta import delta as deltaFunc

from ase.data import covalent_radii
from ase import Atoms
from ase.io import read
from ase.visualize import view

from doubleLJ import doubleLJ_energy_ase as doubleLJ_energy


def num_gradient2(a, delta):
    Natoms = a.get_number_of_atoms()
    dim = 3
    pos0 = a.get_positions()
    dx = 1e-5

    a_down = a.copy()
    a_up = a.copy()
 
    perturb0 = np.zeros((Natoms,dim))
    F_num = np.zeros((Natoms, dim))
    for i in range(Natoms):
        for d in range(dim):
            perturb = perturb0.copy()
            perturb[i,d] = dx/2

            pos_down = pos0 - perturb
            pos_up = pos0 + perturb

            a_down.set_positions(pos_down)
            a_up.set_positions(pos_up)

            E_down = delta.energy(a_down)
            E_up = delta.energy(a_up)

            F_num[i,d] = -(E_up - E_down) / dx
            
    return F_num


r_list = [1.064, 1.4, 1.7, 2.0, 2.2]
X = np.array([[r,0,0,0,0,0] for r in r_list])

dim = 3

L = 2
d = 1
N = 2
pbc = [0,0,0]
atomtypes = ['C', 'C']
atoms_list = []
for x in X:
    positions = x.reshape((-1, dim))
    a = Atoms(atomtypes,
              positions=positions,
              cell=[L,L,d],
              pbc=pbc)
    atoms_list.append(a)

atoms_test = atoms_list[0]
atoms_train = atoms_list[1:]
E_train = np.array([doubleLJ_energy(a) for a in atoms_train])

Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 30
use_angular = False

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)


# Set up KRR-model
comparator = gaussComparator()
delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                delta_function=delta_function,
                bias_fraction=0.7,
                bias_std_add=1)

GSkwargs = {'reg': [1e-5], 'sigma': [0.1]}
MAE, params = krr.train(atoms_list=atoms_train, data_values=E_train, k=2, **GSkwargs)
print(MAE)

E_predict = krr.predict_energy(atoms_test)
F_predict = krr.predict_force(atoms_test)
print(E_predict)
print(F_predict)
print('positions:\n', atoms_test.get_positions())

print(0.7*2*covalent_radii[6])
print(krr.beta)


a0 = read('graphene_data/cand10.traj', index='0')


F0_delta = delta_function.forces(a0).reshape((-1,3))
F0_delta_num = num_gradient2(a0, delta_function)
#print('F0:\n',F0_delta)
#print('F0_num:\n',F0_delta_num)
print('F0_diff:\n',F0_delta - F0_delta_num)


