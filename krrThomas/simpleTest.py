import numpy as np
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator
from krr_class2 import krr_class
from krr_class_new import krr_class as krr_class_new
import time

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


def testModel(model, Ndata, theta=0, new=False):
    Natoms = 4
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    X = createData(Ndata, theta)
    G = model.featureCalculator.get_featureMat(X)
    
    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i, :] = -grad

    # Train model
    t0 = time.time()
    #gridSearch
    GSkwargs = {'reg': np.logspace(-7, -6, 2), 'sigma': np.logspace(0, 2, 3)}
    if new:
        MAE, params = model.train(E, G, **GSkwargs)
    else:
        MAE, params = model.gridSearch(E, G, disp=False, **GSkwargs)
    print('Time used on training:', time.time() - t0)
    
    Npoints = 1000
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Fpredx = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 2*Natoms))

    Gtest = np.zeros((Npoints, G.shape[1]))
    
    delta_array = np.linspace(-3, 3, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 1])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] = pertub_rot

        Gtest[i] = model.featureCalculator.get_singleFeature(Xtest[i])
        
        Etest[i], gradtest = doubleLJ(Xtest[i], eps, r0, sigma)
        Ftest = -gradtest
        Epredict[i] = model.predict_energy(pos=Xtest[i])
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]
        
        Fpred = model.predict_force(pos=Xtest[i])
        Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]

    dx = delta_array[1] - delta_array[0]
    Ffinite = (Epredict[:-1] - Epredict[1:])/dx
    
    return delta_array, Etest, Epredict, Ftestx, Fpredx, Ffinite, Xtest, X
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Natoms = 4
    theta=0*np.pi
    
    # Model 1
    reg = 1e-7  # expKernel: 0.005 , gaussKernel: 1e-7
    sig = 30  # expKernel: 0.3 , gaussKernel: 0.13

    np.random.seed(10)
    featureCalculator = fingerprintFeature(rcut=4)
    comparator = gaussComparator(sigma=sig)
    krr1 = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    print('Model 1')
    t0 = time.time()
    delta_array, Etest1, Epredict1, Ftestx1, Fpredx1, Ffinite1, Xtest, X = testModel(krr1, Ndata=1000, theta=theta, new=False)
    print('Runtime old:', time.time() - t0)
    dx = delta_array[1] - delta_array[0]

    # Model 2
    np.random.seed(10)
    featureCalculator = fingerprintFeature(rcut=4)
    comparator = gaussComparator(sigma=sig)
    krr2 = krr_class_new(comparator=comparator, featureCalculator=featureCalculator)

    print('Model 2')
    t0 = time.time()
    delta_array, Etest2, Epredict2, Ftestx2, Fpredx2, Ffinite2, Xtest, X = testModel(krr2, Ndata=1000, theta=theta, new=True)
    print('Runtime new:', time.time() - t0)
    
    plt.figure(1)
    plt.plot(delta_array, Ftestx1, color='c')
    plt.plot(delta_array, Fpredx1, color='y')
    plt.plot(delta_array, Fpredx2, color='r', linestyle=':')
    #plt.plot(delta_array[1:]-dx/2, Ffinite1, color='g', linestyle=':')

    plt.figure(2)
    plt.plot(delta_array, Etest1, color='c')
    plt.plot(delta_array, Epredict1, color='y')
    plt.plot(delta_array, Epredict2, color='r', linestyle=':')
    
    # Plot first structure
    plt.figure(3)
    plt.plot(Xtest[:, -2], Xtest[:, -1], color='r')
    
    x = X[-1].reshape((Natoms, 2))
    plt.scatter(x[:, 0], x[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    
