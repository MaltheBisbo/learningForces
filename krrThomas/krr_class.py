import numpy as np
import matplotlib.pyplot as plt
from doubleLJ import doubleLJ
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator


class krr_class():
    def __init__(self, featureCalculator=None, comparator=None, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)

        assert self.comparator is not None

    def fit(self, data_values=None, featureMat=None, positionMat=None, lamb=1e-3):
        if data_values is not None:
            self.data_values = data_values

        # calculate features form positions if they are not given
        if featureMat is not None:
            self.featureMat = featureMat
        elif positionMat is not None and self.featureCalculator is not None:
            self.featureObj = self.featureCalculator(positionMat)
            self.featureMat = self.featureObj.get_featureMat()
        else:
            print("You need to set the feature matrix or both the position matrix and a feature calculator")

        self.lamb = lamb
        self.similarityMat = self.comparator.get_similarity_matrix(self.featureMat)

        self.beta = np.mean(data_values)

        A = self.similarityMat + self.lamb*np.identity(self.data_values.shape[0])
        #self.alpha = np.linalg.inv(A).dot(self.data_values-self.beta)
        self.alpha = np.linalg.solve(A, self.data_values - self.beta)
        
    def predict_energy(self, fnew=None, pos=None):
        if fnew is not None:
            self.fnew = fnew
        else:
            self.pos = pos
            assert self.featureCalculator is not None
            self.fnew, self.inew = self.featureCalculator.get_singleFeature(self.pos)
            
        self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        return self.similarityVec.dot(self.alpha) + self.beta

    def predict_force(self, pos=None, fnew=None, inew=None):
        if pos is not None:
            self.pos = pos
        if fnew is not None:
            self.fnew = fnew
            self.similarityVec = self.comparator.get_similarity_vector(self.fnew)
        else:
            assert self.featureCalculator is not None
            self.fnew, self.inew = self.featureCalculator.get_singleFeature(self.pos)
            self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        df_dR = self.featureCalculator.get_featureGradient(self.pos, self.fnew, self.inew)
        dk_df = self.comparator.get_jac(self.fnew)

        kernelDeriv = np.dot(dk_df, df_dR)
        return -(kernelDeriv.T).dot(self.alpha)

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
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 8))
    
    # Define an array of positions for the last pointB
    #xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(0.5, 2, Ndata)
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

    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 100
    lamb = 0.005
    sig = 0.3

    theta = 0.7*np.pi

    X = createData(Ndata, theta)
    featureCalculator = bob_features()
    G = featureCalculator.get_featureMat(X)[0]

    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*5))
    for i in range(Ndata):
        E[i], F[i, :] = doubleLJ(X[i], eps, r0, sigma)
    
    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)

    # Train model
    comparator = eksponentialComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    krr.fit(Etrain, Gtrain, lamb=lamb)

    Npoints = 1000
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Fpredx = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 10))
    print(Xtest.shape)
    delta_array = np.linspace(-3.5, 0.5, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 0])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] += pertub_rot

        Etest[i], Ftest = doubleLJ(Xtest[i], eps, r0, sigma)
        Epredict[i] = krr.predict_energy(pos=Xtest[i])
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]
        
        Fpred = krr.predict_force()
        Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]

    dx = delta_array[1] - delta_array[0]
    Ffinite = (Epredict[:-1] - Epredict[1:])/dx

    plt.figure(1)
    plt.plot(delta_array, Ftestx, color='c')
    plt.plot(delta_array, Fpredx, color='y')
    plt.plot(delta_array[1:], Ffinite, color='g')
    plt.plot(delta_array, Etest)
    plt.plot(delta_array, Epredict, color='r')

    """
    Xtest = Xtest0.copy()
    Xtest[-2] -= 1.2
    gtest, itest = features.calc_singleFeature(Xtest)

    kappaDeriv = kernelVecDeriv(Xtest, gtest, itest, Gtrain, sig)

    a = alpha

    Fpred = -kappaDeriv.dot(a)
    E, Ftest = doubleLJ(Xtest, eps, r0, sigma)
    print(Fpred)
    print(Ftest)
    """

    # Plot first structure
    plt.figure(2)
    plt.scatter(Xtest[:, -2], Xtest[:, -1], color='r')
    plt.scatter(Xtest[0, -2], Xtest[0, -1], color='y')
    
    x = X[-1].reshape((5, 2))
    plt.scatter(x[:, 0], x[:, 1])
    
    plt.show()

    
