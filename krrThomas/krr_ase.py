import numpy as np
import pdb


class krr_class():
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
    def __init__(self, comparator, featureCalculator, reg=1e-5, bias_fraction=0.95, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)
        self.bias_fraction = bias_fraction
        self.reg = reg

        # Initialize data arrays
        max_data = 15000
        length_feature = featureCalculator.Nelements  # featureCalculator.Nbins
        self.data_values = np.zeros(max_data)
        self.featureMat = np.zeros((max_data, length_feature))
        
        # Initialize data counter
        self.Ndata = 0

    def predict_energy(self, atoms=None, fnew=None, similarityVec=None, return_error=False):
        """
        Predict the energy of a new structure.
        """
        if similarityVec is None:
            if fnew is None:
                fnew = self.featureCalculator.get_feature(atoms)
            similarityVec = self.comparator.get_similarity_vector(fnew, self.featureMat[:self.Ndata])

        predicted_value = similarityVec.dot(self.alpha) + self.beta

        if return_error:
            alpha_err = np.dot(self.Ainv, similarityVec)
            #A = self.similarityMat + self.reg*np.identity(self.Ndata)
            #alpha_err = np.linalg.solve(A, similarityVec)
            theta0 = np.dot(self.data_values[:self.Ndata], self.alpha) / self.Ndata
            prediction_error = np.sqrt(np.abs(theta0*(1 - np.dot(similarityVec, alpha_err))))
            return predicted_value, prediction_error, theta0
        else:
            return predicted_value
    
    def predict_force(self, atoms=None, fnew=None, fgrad=None):
        """
        Predict the force of a new structure.
        """
        if fnew is None:
            fnew = self.featureCalculator.get_feature(atoms)
        if fgrad is None:
            fgrad = self.featureCalculator.get_featureGradient(atoms)
        dk_df = self.comparator.get_jac(fnew, featureMat=self.featureMat[:self.Ndata])

        kernelDeriv = np.dot(dk_df, fgrad.T)
        return -(kernelDeriv.T).dot(self.alpha)

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
        
    def __fit(self, data_values, similarityMat, reg):
        """
        Fit the model based on training data.
        - i.e. find the alpha coeficients.
        """
        N_beta = int(self.bias_fraction*len(data_values))
        sorted_data_values = np.sort(data_values)
        sorted_filtered_data_values = sorted_data_values[:N_beta]
        
        self.beta = np.mean(sorted_filtered_data_values)  # + np.var(sorted_filtered_data_values)
        #self.beta = np.mean(data_values)
        
        A = similarityMat + reg*np.identity(len(data_values))
        self.Ainv = np.linalg.inv(A)
        self.alpha = np.dot(self.Ainv, data_values - self.beta)
        #self.alpha = np.linalg.solve(A, data_values - self.beta)
        
    def train(self, atoms_list=None, data_values=None, featureMat=None, add_new_data=True, k=3, **GSkwargs):
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
        
        if featureMat is None:
            featureMat = self.featureCalculator.get_featureMat(atoms_list)
            data_values = np.array([atoms.get_potential_energy() for atoms in atoms_list])
            
        if add_new_data:
            self.add_data(data_values, featureMat)
        else:
            self.Ndata = len(data_values)
            self.data_values[0:self.Ndata] = data_values
            self.featureMat[0:self.Ndata] = featureMat

        FVU, params = self.__gridSearch(self.data_values[:self.Ndata], self.featureMat[:self.Ndata], k=k, **GSkwargs)

        return FVU, params
            
    def __gridSearch(self, data_values, featureMat, k, **GSkwargs):
        """
        Performs grid search in the set of hyperparameters specified in **GSkwargs.

        Used k-fold cross-validation for error estimates.
        """
        sigma_array = GSkwargs['sigma']
        reg_array = GSkwargs['reg']

        FVU_min = None
        best_args = np.zeros(2).astype(int)
        best_similarityMat = None

        for i, sigma in enumerate(sigma_array):
            # Calculate similarity matrix for current sigma
            self.comparator.set_args(sigma=sigma)
            similarityMat = self.comparator.get_similarity_matrix(featureMat)

            for j, reg in enumerate(reg_array):
                FVU = self.__cross_validation(data_values, similarityMat, k=k, reg=reg)
                if FVU_min is None or FVU < FVU_min:
                    FVU_min = FVU
                    best_args = np.array([i, j])
                    best_similarityMat = similarityMat

        self.sigma = sigma_array[best_args[0]]
        self.reg = reg_array[best_args[1]]

        # Set comparator to best sigma
        self.comparator.set_args(sigma=self.sigma)

        # Set similarity matrix to best
        self.similarityMat = best_similarityMat
        
        # Train with best parameters using all data
        self.__fit(data_values, best_similarityMat, reg=self.reg)

        return FVU_min, {'sigma': self.sigma, 'reg': self.reg}
    
    def __cross_validation(self, data_values, similarityMat, k, reg):
        Ndata = len(data_values)

        # Permute data for cross-validation
        permutation = np.random.permutation(Ndata)
        data_values = data_values[permutation]
        similarityMat = similarityMat[:,permutation][permutation,:]
        
        Ntest = int(np.floor(Ndata/k))
        FVU = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata), [ik*Ntest, (ik+1)*Ntest])
            i_train = np.r_[i_train1, i_train2]
            self.__fit(data_values[i_train], similarityMat[i_train,:][:,i_train], reg=reg)

            # Validation
            test_similarities = similarityMat[i_test,:][:,i_train]
            FVU[ik] = self.__get_FVU_energy(data_values[i_test], test_similarities)
        return np.mean(FVU)

    def __get_FVU_energy(self, data_values, test_similarities):
        Epred = self.predict_energy(similarityVec=test_similarities)
        MAE = np.mean(np.abs(Epred - data_values))
        MSE = np.mean((Epred - data_values)**2)
        var = np.var(data_values)
        FVU = MSE / var
        return MAE

