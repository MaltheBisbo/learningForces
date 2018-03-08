import numpy as np
import matplotlib.pyplot as plt

from doubleLJ import doubleLJ_energy, doubleLJ_gradient
from angular_fingerprintFeature_test3 import Angular_Fingerprint
from gaussComparator import gaussComparator
from vector_krr_ase import vector_krr_class

from angular_fingerprintFeature_test import Angular_Fingerprint as Angular_Fingerprint_test
import time
import pdb

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

def createData3d(Ndata, theta, dim=3, pos_start=0, pos_end=1.5):
    # Define fixed points
    x1 = np.array([-1, 0, 1])
    x2 = np.array([0, 0, 0])
    x3 = np.array([0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    x3rot = x3
    xrot = np.c_[x1rot, x2rot, x3rot].reshape((1, dim*x1rot.shape[0]))

    # Define an array of positions for the last point
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(pos_start, pos_end, Ndata)
    x2new = np.ones(Ndata)
    x3new = np.zeros(Ndata)

    # rotate new coordinates
    x1new_rot = np.cos(theta) * x1new - np.sin(theta) * x2new
    x2new_rot = np.sin(theta) * x1new + np.cos(theta) * x2new
    x3new_rot = x3new
    
    xnew_rot = np.c_[x1new_rot, x2new_rot, x3new_rot]

    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(xrot, Ndata, axis=0), xnew_rot]
    return X

def createData2d(Ndata, theta, pos_start=0, pos_end=1.5):
    # Define fixed points
    x1 = np.array([-1, 0, 1])
    x2 = np.array([0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 2*x1rot.shape[0]))

    # Define an array of positions for the last point
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(pos_start, pos_end, Ndata)
    x2new = np.ones(Ndata)

    # rotate new coordinates
    x1new_rot = np.cos(theta) * x1new - np.sin(theta) * x2new
    x2new_rot = np.sin(theta) * x1new + np.cos(theta) * x2new
    
    xnew_rot = np.c_[x1new_rot, x2new_rot]

    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(xrot, Ndata, axis=0), xnew_rot]
    return X, x1new


theta = 0

# Parameters for potential
eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

X_3d = createData3d(100, theta)

dim = 3

L = 2
d = 1
pbc = [0,0,0]
atomtypes = ['H', 'H', 'H', 'H']
cell = [L,L,d]

# Training forces
X_2d_train = createData2d(20, theta, pos_start=-1.5, pos_end=1.5)
Ftrain = -np.array([doubleLJ_gradient(x, eps, r0, sigma)[6] for x in X_2d_train])

# Atoms list for training
X_3d_train = createData3d(20, theta, pos_start=-1.5, pos_end=1.5)
atoms_train = []
for x_test in X_3d_train:
    positions = x_test.reshape((-1,dim))
    a = Atoms(atomtypes,
              positions=positions,
              cell=cell,
              pbc=pbc)
    atoms_train.append(a)

#view(a)

Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 50
use_angular = True

featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

# Set up KRR-model
comparator = gaussComparator()
krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

# Training
MAE, params = krr.train(atoms_list=atoms_train, forces=Ftrain, add_new_data=False)





X_2d, Xpertub = createData2d(100, theta, pos_start=-1.5, pos_end=1.5)
dx = Xpertub[1] - Xpertub[0]
Xpertub_F = Xpertub[:-1]+dx/2

E = np.array([doubleLJ_energy(x, eps, r0, sigma) for x in X_2d])
F = -np.array([doubleLJ_gradient(x, eps, r0, sigma)[6] for x in X_2d])

X_3d_test = createData3d(100, theta, pos_start=-1.5, pos_end=1.5)
atoms_test = []
for x_test in X_3d_test:
    positions = x_test.reshape((-1,dim))
    a = Atoms(atomtypes,
              positions=positions,
              cell=cell,
              pbc=pbc)
    atoms_test.append(a)

Fpred = np.array([krr.predict_force(a)[6] for a in atoms_test])

plt.figure()
plt.plot(Xpertub, E)
plt.plot(Xpertub, F)
plt.plot(Xpertub, Fpred)
plt.show()
