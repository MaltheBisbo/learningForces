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

    def predict_energy(self, fnew=None, pos=None):
        if fnew is not None:
            self.fnew = fnew
        else:
            self.pos = pos
            self.fnew = self.featureCalculator.get_singleFeature(x=self.pos)

        self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        return self.similarityVec.dot(self.alpha) + self.beta

    def predict_force(self, pos, fnew=None):
        if pos is not None:
            self.pos = pos
            self.fnew = self.featureCalculator.get_singleFeature(self.pos)
            self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        df_dR = self.featureCalculator.get_singleGradient(self.pos)
        dk_df = self.comparator.get_jac(self.fnew)

        kernelDeriv = np.dot(dk_df, df_dR)
        return -(kernelDeriv.T).dot(self.alpha)

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

    def get_FVU_force(self, force, positionMat, featureMat=None):
        if featureMat is None:
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        Fpred = np.array([self.predict_force(positionMat[i], featureMat[i])
                          for i in range(force.shape[0])])
        MSE_force = np.mean((Fpred - force)**2, axis=0)
        var_force = np.var(force, axis=0)        
        return MSE_force / var_force
    
