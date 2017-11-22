import numpy as np
import matplotlib.pyplot as plt

import fingerprintFeature
import angular_fingerprintFeature2

from ase import Atoms

def createData(Ndata, theta):
    # Define fixed points
    x1 = np.array([-1, 0, 1])
    x2 = np.array([0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 2*x1rot.shape[0]))

    # Define an array of positions for the last pointB
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(0, 1.5, Ndata)
    x2new = np.ones(Ndata)

    # rotate new coordinates
    x1new_rot = np.cos(theta) * x1new - np.sin(theta) * x2new
    x2new_rot = np.sin(theta) * x1new + np.cos(theta) * x2new

    xnew_rot = np.c_[x1new_rot, x2new_rot]

    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(xrot, Ndata, axis=0), xnew_rot]
    return X

if __name__ == "__main__":
    Ndata = 4
    Natoms = 4
    theta = 0

    X = createData(Ndata, theta)
    xtest = np.array([[0, 0, 0], [0, 1.1, 0], [0.9, 0.4, 0]]) 
    atoms = [Atoms('Au', positions=xtest, pbc=[0,0,0]) for x in X]

    G1 = fingerprintFeature.get_featureMat(X)

    
    
