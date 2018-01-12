import numpy as np
import matplotlib.pyplot as plt

from fingerprintFeature import fingerprintFeature
#from angular_fingerprintFeature2 import Angular_Fingerprint
from angular_fingerprintFeature_m import Angular_Fingerprint
from angular_fingerprintFeature import Angular_Fingerprint as Angular_Fingerprint_tho

from ase import Atoms
from ase.visualize import view

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
    xtest = np.array([(0, 0, 0), (0, 1.1, 0), (1.9, -0.4, 0)])
    atoms = Atoms('H3', xtest)

    rcut = 4
    binwidth = 0.1
    sigma = 0.2
    
    featureCalculator1 = fingerprintFeature(rcut=rcut, binwidth=binwidth, sigma=sigma, dim=3)
    G1 = featureCalculator1.get_singleFeature(xtest.reshape(-1))
    G1_grad = featureCalculator1.get_singleGradient(xtest.reshape(-1))

    Rc1 = 4
    binwidth1 = 0.1
    sigma1 = 0.2
    
    print('\n ase\n')
    featureCalculator2 = Angular_Fingerprint(atoms, Rc1=Rc1, binwidth1=binwidth1, sigma1=sigma1, gamma=30, use_angular=False)
    G2_2body  = featureCalculator2.get_features(atoms)  # *0.75
    G2_grad = featureCalculator2.get_featureGradients(atoms)
    print(G2_grad)

    featureCalculator3 = Angular_Fingerprint_tho(atoms, Rc=Rc1, binwidth1=binwidth1, sigma1=sigma1)
    res3 = featureCalculator3.get_features(atoms)
    G3_2body = np.array(list(res3.values())[0])
    print(G3_2body)
    #view(atoms)

    plt.figure(1)
    plt.plot(np.arange(len(G1)), G1, label='no ase')
    plt.plot(np.arange(len(G2_2body)), G2_2body, label='ase')
    plt.plot(np.arange(len(G3_2body)), G3_2body, label='ase_tho')
    plt.legend()

    plt.figure(2)
    plt.plot(np.arange(len(G2_grad)), G2_grad)
    plt.plot(np.arange(len(G1_grad)), G1_grad, linestyle=':')

    plt.figure(3)
    plt.scatter(xtest[:,0], xtest[:,1])

    plt.show()
    

    
    
    
