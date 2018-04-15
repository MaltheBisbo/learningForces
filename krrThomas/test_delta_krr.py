import numpy as np

from gaussComparator import gaussComparator
#from angular_fingerprintFeature import Angular_Fingerprint
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from delta_functions.delta import delta as deltaFunc

from ase.data import covalent_radii
from ase import Atoms

from doubleLJ import doubleLJ_energy_ase as doubleLJ_energy

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
