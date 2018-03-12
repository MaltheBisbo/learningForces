import numpy as np
from doubleLJ import doubleLJ_energy_ase, doubleLJ_gradient_ase
from angular_fingerprintFeature_test3 import Angular_Fingerprint
from gaussComparator import gaussComparator
from vector_krr_ase import vector_krr_class

import pdb

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

def makeConstrainedStructure2d(Natoms):
    boxsize = 1.5 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 1.5

    def validPosition(X, xnew):
        Natoms = int(len(X)/2)  # Current number of atoms
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[2*i:2*i+2])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    Xinit = np.zeros(2*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.random.rand(2) * boxsize
            if validPosition(Xinit[:2*i], xnew):
                Xinit[2*i:2*i+2] = xnew
                break
    Xinit = Xinit.reshape((-1, 2))
    Xinit = np.c_[Xinit, np.zeros(len(Xinit))]
    return Xinit

def makeConstrainedStructure3d(Natoms):
    dim = 3
    boxsize = 1.5 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 2.2

    def validPosition(X, xnew):
        Natoms = int(len(X)/dim)  # Current number of atoms
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[dim*i:dim*(i+1)])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    Xinit = np.zeros(dim*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.random.rand(dim) * boxsize
            if validPosition(Xinit[:dim*i], xnew):
                Xinit[dim*i:dim*(i+1)] = xnew
                break
    Xinit = Xinit.reshape((-1, 3))
    return Xinit


Nstructures = 50
Natoms = 24

boxsize = 1.5 * np.sqrt(Natoms)
pbc = [0,0,0]
atomtypes = str(Natoms) + 'He'
cell = [boxsize]*3

atoms = []

# Generate structures
for i in range(Nstructures):
    positions = makeConstrainedStructure3d(Natoms)
    a = Atoms(atomtypes,
              positions=positions,
              pbc=pbc,
              cell=cell)
    atoms.append(a)
view(atoms[0])

# Calculate forces
F = np.array([doubleLJ_gradient_ase(a, r0=1.6) for a in atoms])

# Set up featureCalculator
Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 10
use_angular = False

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

# Set up KRR-model
comparator = gaussComparator()
krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

# Training
GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,4,20)}
#GSkwargs = {'reg': [1e-5], 'sigma': [0.2]}
MAE, params = krr.train(atoms_list=atoms, forces=F, add_new_data=False, k=2, **GSkwargs)
print(MAE)
print(params)
