import numpy as np
import pdb


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
            pdb.set_trace()
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
        return np.mean(MAE)

    def __get_MAE_energy(self, forces, kernel_Hess_mat_test):
        Fpred = self.predict_force(kernel_Hess_vec=kernel_Hess_mat_test)
        forces = forces.reshape(-1)
        Fpred = Fpred.reshape(-1)
        MAE = np.mean(np.abs(Fpred - forces))
        #MSE = np.mean((Epred - data_values)**2)
        #var = np.var(data_values)
        #FVU = MSE / var
        return MAE

