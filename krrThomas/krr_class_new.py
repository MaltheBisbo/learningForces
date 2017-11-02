import numpy as np
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from gaussComparator import gaussComparator

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
    def __init__(self, comparator, featureCalculator, reg=1e-5, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)
        self.reg = reg

        # Initialize data arrays
        max_data = 3000
        length_feature = featureCalculator.Nbins
        self.data_values = np.zeros(max_data)
        self.featureMat = np.zeros((max_data, length_feature))

        # Initialize data counter
        self.Ndata = 0
    """
    def fit(self, data_values, featureMat=None, positionMat=None, reg=None):
        self.data_values = data_values

        # Calculate features from positions if they are not given
        if featureMat is not None:
            self.featureMat = featureMat
        else:
            self.featureMat = self.featureCalculator.get_featureMat(positionMat)

        if reg is not None:
            self.reg = reg
        
        self.similarityMat = self.comparator.get_similarity_matrix(self.featureMat)

        self.beta = np.mean(data_values)

        A = self.similarityMat + self.reg*np.identity(self.data_values.shape[0])
        
        self.alpha = np.linalg.solve(A, self.data_values - self.beta)
    """
    def predict_energy(self, fnew=None, pos=None, similarityVec=None):
        """
        Predict the energy of a new structure.
        """
        if similarityVec is None:
            if fnew is not None:
                self.fnew = fnew
            else:
                self.pos = pos
                self.fnew = self.featureCalculator.get_singleFeature(x=self.pos)
            similarityVec = self.comparator.get_similarity_vector(self.fnew, self.featureMat[:self.Ndata])

        return similarityVec.dot(self.alpha) + self.beta

    def predict_force(self, pos, fnew=None):
        """
        Predict the force of a new structure.
        """
        self.pos = pos
        if fnew is None:
            self.fnew = self.featureCalculator.get_singleFeature(self.pos)

        df_dR = self.featureCalculator.get_singleGradient(self.pos)
        dk_df = self.comparator.get_jac(self.fnew)

        kernelDeriv = np.dot(dk_df, df_dR)
        return -(kernelDeriv.T).dot(self.alpha)

    def add_data(self, data_values_add, featureMat_add):
        """ 
        Adds data to previously saved data.
        """
        Nadd = len(data_values_add)
        
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
        self.beta = np.mean(data_values)

        A = similarityMat + reg*np.identity(len(data_values))
        self.alpha = np.linalg.solve(A, data_values - self.beta)
        
    def train(self, data_values, featureMat=None, positionMat=None, add_new_data=True, k=3, **GSkwargs):
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
            featureMat = self.featureCalculator.get_featureMat(positionMat)

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
            self.comparator.set_args(sigma=sigma)
            similarityMat = self.comparator.get_similarity_matrix(featureMat)
            for j, reg in enumerate(reg_array):
                FVU = self.__cross_validation(data_values, similarityMat, k=k, reg=reg)
                if FVU_min is None or FVU < FVU_min:
                    FVU_min = FVU
                    best_args = np.array([i, j])
                    best_similarityMat = similarityMat
        sigma_best = sigma_array[best_args[0]]
        reg_best = reg_array[best_args[1]]

        # Set comparator to best sigma
        self.comparator.set_args(sigma=sigma_best)
        
        # Train with best parameters using all data
        self.__fit(data_values, best_similarityMat, reg=reg_best)

        return FVU_min, {'sigma': sigma_best, 'reg': reg_best}
    
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
            FVU[ik] = self.get_FVU_energy2(data_values[i_test], test_similarities)
        return np.mean(FVU)
    """    
    def cross_validation(self, data_values, featureMat, k=3, reg=None):
        Ndata = data_values.shape[0]
        permutation = np.random.permutation(Ndata)
        data_values = data_values[permutation]
        featureMat = featureMat[permutation]

        Ntest = int(np.floor(Ndata/k))
        FVU = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata),
                                                    [Ntest * ik, Ntest * (ik+1)])
            i_train = np.r_[i_train1, i_train2]
            self.fit(data_values[i_train], featureMat[i_train], reg=reg)
            FVU[ik] = self.get_FVU_energy(data_values[i_test], featureMat[i_test])
        return np.mean(FVU)

    def cross_validation_EandF(self, energy, force, featureMat, positionMat, k=3, reg=None):
        Ndata, Ndf = positionMat.shape
        permutation = np.random.permutation(Ndata)
        energy = energy[permutation]
        force = force[permutation]
        featureMat = featureMat[permutation]
        positionMat = positionMat[permutation]
        
        Ntest = int(np.floor(Ndata/k))
        FVU_energy = np.zeros(k)
        FVU_force = np.zeros((k, Ndf))
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata),
                                                    [Ntest * ik, Ntest * (ik+1)])
            #print('index:', ik)
            i_train = np.r_[i_train1, i_train2]
            self.fit(energy[i_train], featureMat[i_train], reg=reg)
            FVU_energy[ik] = self.get_FVU_energy(energy[i_test], featureMat[i_test])
            FVU_force[ik, :] = self.get_FVU_force(force[i_test], positionMat[i_test],
                                                  featureMat[i_test])
        return np.mean(FVU_energy), np.mean(FVU_force, axis=0)

    def gridSearch(self, data_values, featureMat=None, positionMat=None, k=3, disp=False, **GSkwargs):
        if featureMat is None:
            featureMat = self.featureCalculator.get_featureMat(positionMat)

        sigma_array = GSkwargs['sigma']
        reg_array = GSkwargs['reg']
        Nsigma = len(sigma_array)
        Nreg = len(reg_array)
        best_args = np.zeros(2).astype(int)
        FVU_min = None
        for i in range(Nsigma):
            self.comparator.set_args(sigma=sigma_array[i])
            for j in range(Nreg):
                FVU = self.cross_validation(data_values, featureMat, k=k, reg=reg_array[j])
                if disp:
                    print('FVU:', FVU,'params: (', sigma_array[i],',', reg_array[j], ')')
                if FVU_min is None or FVU < FVU_min:
                    FVU_min = FVU
                    best_args = np.array([i, j])
        sigma_best = sigma_array[best_args[0]]
        reg_best = reg_array[best_args[1]]
        # Train with best parameters using all data
        self.comparator.set_args(sigma=sigma_best)
        self.fit(data_values, featureMat, reg=reg_best)
        return FVU_min, {'sigma': sigma_best, 'reg': reg_best}

    def gridSearch_EandF(self, energy, force, featureMat=None, positionMat=None, k=3, disp=False, **GSkwargs):
        if positionMat is not None and self.featureCalculator is not None:
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        else:
            assert featureMat is not None
        sigma_array = GSkwargs['sigma']
        reg_array = GSkwargs['reg']
        Nsigma = len(sigma_array)
        Nreg = len(reg_array)
        best_args = np.zeros(2).astype(int)
        FVU_energy_min = None
        FVU_force_min = None
        for i in range(Nsigma):
            self.comparator.set_args(sigma=sigma_array[i])
            for j in range(Nreg):
                FVU_energy, FVU_force = self.cross_validation_EandF(energy, force, featureMat, positionMat, k=k, reg=reg_array[j])
                if disp:
                    print('FVU:', FVU,'params: (', sigma_array[i],',', reg_array[j], ')')
                if FVU_energy_min is None or FVU_energy < FVU_energy_min:
                    FVU_energy_min = FVU_energy
                    FVU_force_min = FVU_force
                    best_args = np.array([i, j])
        sigma_best = sigma_array[best_args[0]]
        reg_best = reg_array[best_args[1]]
        # Train with best parameters using all data
        self.comparator.set_args(sigma=sigma_best)
        self.fit(energy, featureMat, reg=reg_best)
        return FVU_energy_min, FVU_force_min, {'sigma': sigma_best, 'reg': reg_best}

    def get_FVU_energy(self, data_values, featureMat=None, positionMat=None):
        if featureMat is None:
            assert positionMat is not None
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        Epred = np.array([self.predict_energy(f) for f in featureMat])
        error = Epred - data_values
        MSE = np.mean((Epred - data_values)**2)
        var = np.var(data_values)
        return MSE / var
    """
    def get_FVU_energy2(self, data_values, test_similarities):
        Epred = np.array([self.predict_energy(similarityVec=similarity) for similarity in test_similarities])
        error = Epred - data_values
        MSE = np.mean((Epred - data_values)**2)
        var = np.var(data_values)
        return MSE / var
    """
    def get_FVU_force(self, force, positionMat, featureMat=None):
        if featureMat is None:
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        Fpred = np.array([self.predict_force(positionMat[i], featureMat[i])
                          for i in range(force.shape[0])])
        MSE_force = np.mean((Fpred - force)**2, axis=0)
        var_force = np.var(force, axis=0)        
        return MSE_force / var_force
    """
