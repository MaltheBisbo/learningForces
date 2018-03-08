import numpy as np
import pdb

from doubleLJ import doubleLJ_energy, doubleLJ_gradient
from doubleLJ import doubleLJ
from angular_fingerprintFeature_test3 import Angular_Fingerprint
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator


class vector_krr_class():
    """
    comparator:
    Class to calculate similarities between structures based on their feature vectors.
    The comparator coresponds to the choice of kernel for the model

    featureCalculator:
    Class to calculate the features of structures based on the atomic positions of the structures.

    reg:
    Regularization parameter for the model

    comparator_kwargs:
    Parameters for the compator. This could be the width for the gaussian kernel.
    """
    def __init__(self, comparator, featureCalculator, reg=1e-5, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)
        self.reg = reg

        # Initialize data arrays
        max_data = 15000
        length_feature = featureCalculator.Nelements  # featureCalculator.Nbins
        self.Natoms = featureCalculator.n_atoms
        self.dim = featureCalculator.dim
        self.Ncoord = self.Natoms*self.dim

        self.forces = np.zeros((max_data, self.Ncoord))
        self.featureMat = np.zeros((max_data, length_feature))
        self.featureGradMat = np.zeros((max_data, self.Ncoord, length_feature))
        
        # Initialize data counter
        self.Ndata = 0

    def predict_energy(self, atoms=None, fnew=None):
        """
        Predict the energy of a new structure.
        """
        if fnew is None:
            fnew = self.featureCalculator.get_feature(atoms)

        kernel_Jac = self.comparator.get_jac_new(fnew, self.featureMat)

        kernel_Jac_vec = np.zeros((1,self.Ncoord*self.Ndata))
        for i in range(self.Ndata):
            kernel_Jac_vec[:, i*self.Ncoord:(i+1)*self.Ncoord] = kernel_Jac[i, :] @ self.featureGradMat[i].T

        return -kernel_Jac_vec @ self.alpha
    
    def predict_force(self, atoms=None, fnew=None, fnew_grad=None, kernel_Hess_vec=None):
        """
        Predict the force of a new structure.
        """
        
        if kernel_Hess_vec is None:
            if fnew is None:
                fnew = self.featureCalculator.get_feature(atoms)
            if fnew_grad is None:
                fnew_grad = self.featureCalculator.get_featureGradient(atoms)
            
            kernel_Hess_vec = np.zeros((self.Ncoord, self.Ncoord*self.Ndata))
            for j in range(self.Ndata):
                kernel_Hess = self.comparator.get_single_Hess(fnew, self.featureMat[j])
                kernel_Hess_vec[:, j*self.Ncoord:(j+1)*self.Ncoord] = fnew_grad @ kernel_Hess @ self.featureGradMat[j].T

        return kernel_Hess_vec @ self.alpha

    def add_data(self, data_values_add, featureMat_add):
        """
        Adds data to previously saved data.
        """
        Nadd = len(data_values_add)

        if Nadd > 0:
            # Add data
            self.data_values[self.Ndata:self.Ndata+Nadd] = data_values_add
            self.featureMat[self.Ndata:self.Ndata+Nadd] = featureMat_add
            
            # Iterate data counter
            self.Ndata += Nadd

    def remove_data(self, N_remove):
        """
        Removes the N_remove oldest datapoints
        """
        # Remove data
        self.data_values[0:self.Ndata-N_remove] = self.data_values[N_remove:self.Ndata]
        self.featureMat[0:self.Ndata-N_remove] = self.featureMat[N_remove:self.Ndata]

        # Adjust data counter
        self.Ndata -= N_remove
        
    def __fit(self, forces, kernel_Hess_mat, reg):
        """
        Fit the model based on training data.
        - i.e. find the alpha coeficients.
        """
        Ndata_fit = len(forces)
        forces = forces.reshape(-1)
        A = kernel_Hess_mat - self.reg*np.identity(Ndata_fit*self.Ncoord)
        self.alpha = np.linalg.solve(A, forces)
        
    def train(self, atoms_list=None, forces=None, featureMat=None, featureGradMat=None, add_new_data=True, k=3, **GSkwargs):
        """
        Train the model using gridsearch and cross-validation
            
        --- Input ---
        data_values:
        The labels of the new training data. In our case, the energies of the new training structures.

        featureMat:
        The features of the new training structures.

        positionMat:
        The atomic positions of the new training structures.

        add_new_data:
        If True, the data passed will be added to previously saved data (if any).

        k:
        Performs k-fold cross-validation.

        **GSkwargs:
        Dict containing the sequences of the kernel-width and regularization parameter to be
        used in grissearch. The labels are 'sigma' and 'reg' respectively.
        """
        Ncoord = self.Natoms*self.dim
        
        if featureMat is None:
            featureMat = self.featureCalculator.get_featureMat(atoms_list)
        if featureGradMat is None:
            featureGradMat = self.featureCalculator.get_all_featureGradients(atoms_list)

        if forces is None:
            forces = np.array([atoms.get_forces() for atoms in atoms_list])
            
        if add_new_data:
            pass
            # self.add_data(data_values, featureMat)
        else:
            self.Ndata = len(forces)
            self.forces[0:self.Ndata] = forces.reshape((self.Ndata, Ncoord))
            self.featureMat[0:self.Ndata] = featureMat
            self.featureGradMat[0:self.Ndata] = featureGradMat

        FVU, params = self.__gridSearch(self.forces[:self.Ndata], self.featureMat[:self.Ndata],
                                        self.featureGradMat[0:self.Ndata], k=k, **GSkwargs)

        return FVU, params

    def __calculate_kernelHess(self, featureMat, featureGradMat, sigma):
        # Set sigma
        self.comparator.set_args(sigma=sigma)

        Ncoord = self.Natoms*self.dim
        kernel_Hess_mat = np.zeros((Ncoord*self.Ndata, Ncoord*self.Ndata))
        for n in range(self.Ndata):
            for m in range(self.Ndata):
                kernel_Hess = self.comparator.get_single_Hess(featureMat[n], featureMat[m])
                kernel_Hess_mat[n*Ncoord:(n+1)*Ncoord,
                                m*Ncoord:(m+1)*Ncoord] = featureGradMat[n] @ kernel_Hess @ featureGradMat[m].T
        return kernel_Hess_mat
    
    def __gridSearch(self, forces, featureMat, featureGradMat, k, **GSkwargs):
        """
        Performs grid search in the set of hyperparameters specified in **GSkwargs.

        Used k-fold cross-validation for error estimates.
        """
        
        sigma_array = GSkwargs['sigma']
        reg_array = GSkwargs['reg']

        MAE_min = None
        best_args = np.zeros(2).astype(int)
        best_kernel_Hess_mat = None
        
        for i, sigma in enumerate(sigma_array):
            # Calculate matrix of Hessians
            kernel_Hess_mat = self.__calculate_kernelHess(featureMat, featureGradMat, sigma)

            for j, reg in enumerate(reg_array):
                MAE = self.__cross_validation(forces, kernel_Hess_mat, k=k, reg=reg)
                if MAE_min is None or MAE < MAE_min:
                    MAE_min = MAE
                    best_args = np.array([i, j])
                    best_kernel_Hess_mat = kernel_Hess_mat

        self.sigma = sigma_array[best_args[0]]
        self.reg = reg_array[best_args[1]]

        # Set comparator to best sigma
        self.comparator.set_args(sigma=self.sigma)

        # Set similarity matrix to best
        self.kernel_Hess_mat = best_kernel_Hess_mat
        
        # Train with best parameters using all data
        self.__fit(forces, best_kernel_Hess_mat, reg=self.reg)

        return MAE_min, {'sigma': self.sigma, 'reg': self.reg}

    def __permute_kernel_Hess_mat(self, kernel_Hess_mat, permutation):
        Ndata = self.Ndata
        Ncoord = self.Natoms*self.dim

        Hess_permutation = np.repeat(Ncoord*permutation, Ncoord) + np.tile(np.arange(Ncoord), Ndata)
        return kernel_Hess_mat[Hess_permutation, :][:, Hess_permutation]

    def __subset_kernel_Hess_mat(self, kernel_Hess_mat, indices1, indices2):
        Ncoord = self.Natoms*self.dim
        Ndata1 = len(indices1)
        Ndata2 = len(indices2)

        Hess_indices1 = np.repeat(Ncoord*indices1, Ncoord) + np.tile(np.arange(Ncoord), Ndata1)
        Hess_indices2 = np.repeat(Ncoord*indices2, Ncoord) + np.tile(np.arange(Ncoord), Ndata2)
        return kernel_Hess_mat[Hess_indices1, :][:, Hess_indices2]
        
    def __cross_validation(self, forces, kernel_Hess_mat, k, reg):
        Ndata = self.Ndata

        # Permute data for cross-validation
        permutation = np.random.permutation(Ndata)
        forces = forces[permutation]
        kernel_Hess_mat = self.__permute_kernel_Hess_mat(kernel_Hess_mat, permutation)
        
        Ntest = int(np.floor(Ndata/k))
        MAE = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata), [ik*Ntest, (ik+1)*Ntest])
            i_train = np.r_[i_train1, i_train2]

            # Training
            kernel_Hess_mat_train = self.__subset_kernel_Hess_mat(kernel_Hess_mat, i_train, i_train)
            self.__fit(forces[i_train], kernel_Hess_mat_train, reg=reg)

            # Validation
            kernel_Hess_mat_test = self.__subset_kernel_Hess_mat(kernel_Hess_mat, i_test, i_train)
            MAE[ik] = self.__get_MAE_energy(forces[i_test], kernel_Hess_mat_test)
        pdb.set_trace()
        return np.mean(MAE)

    def __get_MAE_energy(self, forces, kernel_Hess_mat_test):
        Fpred = self.predict_force(kernel_Hess_vec=kernel_Hess_mat_test)
        forces = forces.reshape(-1)
        Fpred = Fpred.reshape(-1)
        MAE = np.mean(np.abs(Fpred - forces))
        MSE = np.mean((Fpred - forces)**2)
        var = np.var(forces)
        FVU = MSE / var
        #pdb.set_trace()
        return FVU

def createData(Ndata, theta):
    # Define fixed points
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 8))

    # Define an array of positions for the last pointB
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
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

    Natoms = 5
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 6
    reg = 1e-7  # 1e-7
    sig = 3.5  # 0.13

    theta = 0.1*np.pi

    X = createData(Ndata, theta)
    featureCalculator = fingerprintFeature()
    G = featureCalculator.get_featureMat(X)
    
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
    krr = vector_krr_class(comparator=comparator, featureCalculator=featureCalculator)

    GSkwargs = {'sigma': np.logspace(-2,2,20), 'reg': [1e-7]}
    FVU, params = krr.train(F, X, **GSkwargs)
    
    #krr.fit(Ftrain, Xtrain, reg=reg)

    Npoints = 1000
    Epred = np.zeros(Npoints)
    Fpredx = np.zeros(Npoints)
    Etest = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 2*Natoms))
    print(Xtest.shape)
    # delta_array = np.linspace(-3.5, 0.5, Npoints)
    delta_array = np.linspace(-4, 1, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 0])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] += pertub_rot

        Etest[i], Ftest = doubleLJ(Xtest[i], eps, r0, sigma)
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]
    
        Fpred = krr.predict_force(Xtest[i])
        Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]
        Epred[i] = krr.predict_energy(Xtest[i])
    plt.figure(1)
    plt.plot(delta_array, Ftestx, color='c')
    plt.plot(delta_array, Fpredx, color='y')
    plt.plot(delta_array, Etest, color='b')
    plt.plot(delta_array, Epred, color='r')
    
    # Plot first structure
    plt.figure(2)
    #plt.scatter(Xtest[:, -2], Xtest[:, -1], color='r')
    plt.plot(Xtest[:, -2], Xtest[:, -1], color='r')
    plt.scatter(Xtest[0, -2], Xtest[0, -1], color='y')
    
    x = X[-1].reshape((Natoms, 2))
    plt.scatter(x[:, 0], x[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
