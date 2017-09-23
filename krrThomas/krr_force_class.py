import numpy as np
import matplotlib.pyplot as plt
from doubleLJ import doubleLJ
from bob_features import bob_features
from gaussComparator import gaussComparator


class krr_force_class():
    """
    Vectorized kernel ridge regression:
    Predicts forces by training directly on the forces using the scheme
    described in: www.ncbi.nlm.nih.gov/pmc/articles/PMC5419702/
    
    -- Input --
    featureCalculator:
    Class to calculate the feature of a structure based on atomic positions.

    comparator:
    Class to calculate the similarity of the kernel of the features.
    Must include method to calculate the Hessian of the kernel between
    two features

    **comparator_kwargs:
    parameters for the comparator class (ie. the kernel)
    """
    def __init__(self, featureCalculator=None, comparator=None, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)

        assert self.comparator is not None

    def fit(self, forceMat, positionMat, featureMat=None, lamb=1e-3):
        (Ndata, Ncoord) = positionMat.shape  # Ncoord is number of coordinated in a structure
        self.forceVec = forceMat.reshape(Ndata*Ncoord, order='F')
        self.positionMat = positionMat
    
        if featureMat is not None:
            self.featureMat = featureMat
        elif positionMat is not None and self.featureCalculator is not None:
            self.featureMat, self.indexMat = self.featureCalculator.get_featureMat(positionMat)
        else:
            print("You need to set the feature matrix or both the position matrix and a feature calculator")

        self.lamb = lamb
        #self.similarityMat = self.comparator.get_similarity_matrix(self.featureMat)
        
        # Calculate the matrix consisting of the
        # Hessians of the kernel with respect to
        # the atomic coordinates of two structures
        self.featureGrad = np.array([self.featureCalculator.get_featureGradient(self.positionMat[i],
                                                                                self.featureMat[i],
                                                                                self.indexMat[i])
                                     for i in range(Ndata)])

        kernel_Hess_mat = np.zeros((Ncoord*Ndata, Ncoord*Ndata))
        for i in range(Ndata):
            for j in range(Ndata):
                kernel_Hess = self.comparator.get_Hess_single(self.featureMat[i], self.featureMat[j])
                kernel_Hess_mat[i*Ncoord:(i+1)*Ncoord,
                                j*Ncoord:(j+1)*Ncoord] = self.featureGrad[i].T @ kernel_Hess @ self.featureGrad[j]

        A = kernel_Hess_mat - self.lamb*np.identity(Ndata*Ncoord)
        self.alpha = np.linalg.solve(A, self.forceVec)

    def predict_force(self, pos_new, fnew=None, inew=None):
        if fnew is None or inew is None:
            assert self.featureCalculator is not None
            fnew, inew = self.featureCalculator.get_singleFeature(pos_new)

        (Ndata, Ncoord) = self.positionMat.shape
        featureGrad_new = self.featureCalculator.get_featureGradient(pos_new, fnew, inew)

        kernel_Hess_vec = np.zeros((Ncoord, Ncoord*Ndata))
        for j in range(Ndata):
            kernel_Hess = self.comparator.get_Hess_single(fnew, self.featureMat[j])
            kernel_Hess_vec[:, j*Ncoord:(j+1)*Ncoord] = featureGrad_new.T @ kernel_Hess @ self.featureGrad[j]

        return kernel_Hess_vec @ self.alpha

    def cross_validation(self, data_values, featureMat, k=3, lamb=None, **GSkwargs):
        Ndata = data_values.shape[0]
        permutation = np.random.permutation(Ndata)
        data_values = data_values[permutation]
        featureMat = featureMat[permutation]

        Ntest = int(np.floor(Ndata/k))
        MAE = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata),
                                                    [Ntest * ik, Ntest * (ik+1)])
            i_train = np.r_[i_train1, i_train2]
            self.fit(data_values[i_train], featureMat[i_train], lamb=lamb)
            MAE[ik] = self.get_MAE_energy(data_values[i_test], featureMat[i_test])
        return np.mean(MAE)

    def gridSearch(self, data_values, featureMat, k=3, **GSkwargs):
        sigma_array = GSkwargs['sigma']
        lamb_array = GSkwargs['lamb']
        Nsigma = len(sigma_array)
        Nlamb = len(lamb_array)
        best_args = np.zeros(2).astype(int)
        MAE_min = None
        for i in range(Nsigma):
            self.comparator.set_args(sigma=sigma_array[i])
            for j in range(Nlamb):
                MAE = self.cross_validation(data_values, featureMat, k=k, lamb=lamb_array[j])
                print('MAE:', MAE,'params: (', sigma_array[i],',', lamb_array[j], ')')
                if MAE_min is None or MAE < MAE_min:
                    MAE_min = MAE
                    best_args = np.array([i, j])
        sigma_best = sigma_array[best_args[0]]
        lamb_best = lamb_array[best_args[1]]
        # Train with best parameters using all data
        self.comparator.set_args(sigma=sigma_best)
        self.fit(data_values, featureMat, lamb=lamb_best)
        return MAE_min, {'sigma': sigma_best, 'lamb': lamb_best}

    def get_MAE_energy(self, data_values, featureMat):
        Epred = np.array([self.predict_energy(f) for f in featureMat])
        MAE = np.mean(np.fabs(Epred - data_values))
        return MAE


def createData(Ndata, theta):
    # Define fixed points
    x1 = np.array([-1, 0, 1])
    x2 = np.array([0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 6))

    # Define an array of positions for the last pointB
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(0, 1, Ndata)
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

    Natoms = 4
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 4
    lamb = 1e-7
    sig = 0.13

    theta = 0.7*np.pi

    X = createData(Ndata, theta)
    featureCalculator = bob_features()
    G = featureCalculator.get_featureMat(X)[0]
    
    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], F[i, :] = doubleLJ(X[i], eps, r0, sigma)

    Xtrain = X[:-1]
    Gtrain = G[:-1]
    Ftrain = F[:-1]

    # Train model
    comparator = gaussComparator(sigma=sig)
    krr = krr_force_class(comparator=comparator, featureCalculator=featureCalculator)
    krr.fit(Ftrain, Xtrain, lamb=lamb)

    Npoints = 1000
    Fpredx = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 2*Natoms))
    print(Xtest.shape)
    # delta_array = np.linspace(-3.5, 0.5, Npoints)
    delta_array = np.linspace(-1, 0, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 0])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] += pertub_rot

        Etemp, Ftest = doubleLJ(Xtest[i], eps, r0, sigma)
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]

        Fpred = krr.predict_force(Xtest[i])
        Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]

    plt.figure(1)
    plt.plot(delta_array, Ftestx, color='c')
    plt.scatter(delta_array, Fpredx, color='y')

    # Plot first structure
    plt.figure(2)
    plt.scatter(Xtest[:, -2], Xtest[:, -1], color='r')
    plt.scatter(Xtest[0, -2], Xtest[0, -1], color='y')
    
    x = X[-1].reshape((Natoms, 2))
    plt.scatter(x[:, 0], x[:, 1])
    
    plt.show()

    
