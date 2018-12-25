import numpy as np
import pdb


class GPR():
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
    def __init__(self, comparator, featureCalculator, delta_function=None, bias_func=None, regGS=None, sigmaGS=None):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.regGS = regGS
        self.sigmaGS = sigmaGS
        self.bias_func = bias_func

        self.delta_function=delta_function

        # Initialize data counter
        self.Ndata = 0

    def predict_energy(self, atoms=None, fnew=None, similarityVec=None, delta_values=None, return_error=False, no_prior_on_error=False):
        """
        Predict the energy of a new structure.
        """
        if similarityVec is None:
            if fnew is None:
                fnew = self.featureCalculator.get_feature(atoms)
            similarityVec = self.comparator.get_similarity_vector(fnew, self.featureMat)

        if delta_values is None:
            if self.delta_function is not None:
                delta = self.delta_function.energy(atoms)
            else:
                delta = 0
        else:
            delta = delta_values
            
        predicted_value = similarityVec.dot(self.alpha) + self.bias + delta

        if return_error:
            alpha_err = np.dot(self.Ainv, similarityVec)
            prediction_error = np.sqrt(np.abs(self.theta0*(1 - np.dot(similarityVec, alpha_err))))
            return predicted_value, prediction_error, self.theta0
        else:
            return predicted_value
    
    def predict_force(self, atoms=None, fnew=None, fgrad=None, return_error=False):
        """
        Predict the force of a new structure.
        """

        # Calculate features and their gradients if not given
        if fnew is None:
            fnew = self.featureCalculator.get_feature(atoms)
        if fgrad is None:
            fgrad = self.featureCalculator.get_featureGradient(atoms)
        dk_df = self.comparator.get_jac(fnew, featureMat=self.featureMat)
        
        # Calculate contribution from delta-function
        if self.delta_function is not None:
            delta_force = self.delta_function.forces(atoms)
        else:
            Ncoord = 3 * atoms.get_number_of_atoms()
            delta_force = np.zeros(Ncoord)

        if return_error:
            similarityVec = self.comparator.get_similarity_vector(fnew, self.featureMat)
            alpha_err = np.dot(self.Ainv, similarityVec)
            g = self.theta0*(1 - np.dot(similarityVec, alpha_err))
            kernelDeriv = np.dot(dk_df, fgrad.T)
            error_force = -self.theta0/np.sqrt(g) * (kernelDeriv.T).dot(alpha_err)
            return  -(kernelDeriv.T).dot(self.alpha) + delta_force, error_force
        else:
            kernelDeriv = np.dot(dk_df, fgrad.T)
            return -(kernelDeriv.T).dot(self.alpha) + delta_force

    def save_data(self, data_values_save, featureMat_save, delta_values_save=None, add_new_data=False):
        """
        Adds data to previously saved data.
        """
        Nsave = len(data_values_save)

        if Nsave > 0:
            if add_new_data and self.Ndata > 0:
                # Add data
                self.data_values = np.r_[self.data_values, data_values_save]
                self.featureMat = np.r_[self.featureMat, featureMat_save]
                if self.delta_function is not None:
                    self.delta_values = np.r_[self.delta_values, delta_values_save]
            else:
                # Initialize data objects
                self.Ndata = len(data_values_save)
                self.data_values = data_values_save
                self.featureMat = featureMat_save
                if self.delta_function is not None:
                    self.delta_values = delta_values_save
            # Iterate data counter
            self.Ndata += Nsave

    def calc_bias(self, data_values):
        if self.bias_func is not None:
            if callable(self.bias_func):
                self.bias = self.bias_func(data_values)
            else:
                self.bias = self.bias_func
        else:
            self.bias = 0
        
    def __fit(self, data_values, similarityMat, reg, delta_values=None):
        """
        Fit the model based on training data.
        - i.e. find the alpha coeficients.
        """
        self.calc_bias(data_values)

        if delta_values is None:
            delta_values = 0
        A = similarityMat + reg*np.identity(len(data_values))
        self.Ainv = np.linalg.inv(A)
        self.alpha = np.dot(self.Ainv, data_values - delta_values - self.bias)
        
    def train(self, atoms_list=None, data_values=None, features=None, delta_values=None, add_new_data=True, k=3,
              regGS=None, sigmaGS=None):
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
        
        if regGS is not None:
            self.regGS = regGS
        assert self.regGS is not None
        
        if features is None:
            features = self.featureCalculator.get_featureMat(atoms_list)
            
        if data_values is None:
            data_values = np.array([atoms.get_potential_energy() for atoms in atoms_list])

        if delta_values is None:
            if self.delta_function is not None:
                delta_values = np.array([self.delta_function.energy(a) for a in atoms_list])

        self.save_data(data_values_save=data_values,
                       featureMat_save=features,
                       delta_values_save=delta_values,
                       add_new_data=add_new_data)

        if self.delta_function is not None:
            delta_values_all = self.delta_values
        else:
            delta_values_all = None
        
        FVU, params = self.__gridSearch(self.data_values,
                                        self.featureMat,
                                        k=k,
                                        delta_values=delta_values_all)

        # Calculate theta0 for error estimates
        #self.theta0 = np.dot(self.data_values, self.alpha) / self.Ndata  # old
        self.calc_bias(self.data_values)
        if self.delta_function is not None:
            self.theta0 = np.dot(self.data_values - self.delta_values - self.bias, self.alpha) / self.Ndata
        else:
            self.theta0 = np.dot(self.data_values - self.bias, self.alpha) / self.Ndata
            
        return FVU, params
            
    def __gridSearch(self, data_values, featureMat, k, delta_values=None):
        """
        Performs grid search in the set of hyperparameters specified in **GSkwargs.

        Used k-fold cross-validation for error estimates.
        """
        sigma_array = self.sigmaGS
        reg_array = self.regGS

        self.reg = reg_array[0]
        similarityMat = self.comparator.get_similarity_matrix(featureMat)
        self.__fit(data_values, similarityMat, reg=self.reg, delta_values=delta_values)
        return None, {'reg': self.reg}

    def __cross_validation(self, data_values, similarityMat, k, reg, delta_values=None):
        Ndata = len(data_values)

        # Permute data for cross-validation
        permutation = np.random.permutation(Ndata)
        data_values = data_values[permutation]
        if delta_values is not None:
            delta_values = delta_values[permutation]
        similarityMat = similarityMat[:,permutation][permutation,:]
        
        Ntest = int(np.floor(Ndata/k))
        FVU = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata), [ik*Ntest, (ik+1)*Ntest])
            i_train = np.r_[i_train1, i_train2]
            if delta_values is not None:
                delta_values_train = delta_values[i_train]
                delta_values_test = delta_values[i_test]
            else:
                delta_values_train = None
                delta_values_test = None

            self.__fit(data_values[i_train],
                       similarityMat[i_train,:][:,i_train],
                       reg=reg,
                       delta_values=delta_values_train)

            # Validation
            test_similarities = similarityMat[i_test,:][:,i_train]
            FVU[ik] = self.__get_FVU_energy(data_values[i_test],
                                            test_similarities,
                                            delta_values_test)
        return np.mean(FVU)

    def __get_FVU_energy(self, data_values, test_similarities, delta_values=None):
        Epred = self.predict_energy(similarityVec=test_similarities,
                                    delta_values=delta_values)
        MAE = np.mean(np.abs(Epred - data_values))
        MSE = np.mean((Epred - data_values)**2)
        var = np.var(data_values)
        FVU = MSE / var
        return MAE

if __name__ == '__main__':
    from ase.io import read
    from gaussComparator import gaussComparator
    from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
    from ase.visualize import view
    from ase import Atoms

    from custom_calculators import doubleLJ_calculator

    import matplotlib.pyplot as plt

    def finite_diff(krr, a, dx=1e-5, with_ud=False):
        pos0 = a.get_positions()
        Natoms, dim = pos0.shape
        F = np.zeros((Natoms, dim))
        vu = np.zeros((Natoms, dim))
        vd = np.zeros((Natoms, dim))
        for i in range(Natoms):
            for j in range(dim):
                pos_up = np.copy(pos0)
                pos_up[i,j] += dx/2
                pos_down = np.copy(pos0)
                pos_down[i,j] -= dx/2

                
                a_up = a.copy()
                a_down = a.copy()
                a_up.set_positions(pos_up)
                a_down.set_positions(pos_down)
                
                E_up, err_up, _ = krr.predict_energy(a_up, return_error=True)
                val_up = E_up - err_up
                E_down, err_down, _ = krr.predict_energy(a_down, return_error=True)
                val_down = E_down - err_down
                #print(E_down, E_up)
                #print(err_down, err_up)
                #print('val:', val_down - val_up)
                #print('E  :', E_down - E_up)

                vu[i,j] = val_up
                vd[i,j] = val_down
                
                F[i,j] = (val_down - val_up)/dx
        if with_ud:
            return F[0,0], vu[0,0], vd[0,0]
        else:
            return F
    
    def createData(r):
        positions = np.array([[0,0,0],[r,0,0]])
        a = Atoms('2H', positions, cell=[3,3,1], pbc=[0,0,0])
        calc = doubleLJ_calculator()
        a.set_calculator(calc)
        return a

    def test1():
        a_train = [createData(r) for r in [0.9,1,1.3,2,3]]
        
        #traj = read('graphene_data/all_every10th.traj', index='0::5')
        #a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = a_train[0].get_number_of_atoms()
        #view(a_train)
        
        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        
        # Set up KRR-model
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        Ntest = 100
        r_test = np.linspace(0.87, 3.5, Ntest)
        E_test = np.zeros(Ntest)
        err_test = np.zeros(Ntest)
        F_test = np.zeros(Ntest)
        E_true = np.zeros(Ntest)
        F_true = np.zeros(Ntest)
        F_num = np.zeros(Ntest)
        for i, r in enumerate(r_test):
            ai = createData(r)
            E, err, _ = krr.predict_energy(ai, return_error=True)
            E_test[i] = E
            err_test[i] = err
            
            F_test[i] = krr.predict_force(ai, with_error=True)[0]
            F_num[i] = finite_diff(krr, ai)[0,0]
            
            
            E_true[i] = ai.get_potential_energy()
            F_true[i] = ai.get_forces()[0,0]
            
            
        plt.figure()
        plt.plot(r_test, E_true, label='true')
        plt.plot(r_test, E_test, label='model')
        plt.plot(r_test, E_test-err_test, label='model')
        plt.legend()
        
        plt.figure()
        plt.plot(r_test, F_true, label='true')
        plt.plot(r_test, F_test, label='model')
        plt.plot(r_test, F_num, 'k:', label='num')
        plt.legend()

    
    def test2():
        traj = read('graphene_data/all_every10th.traj', index='0::5')
        a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = a_train[0].get_number_of_atoms()

        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        a_test = traj[99]
        print(krr.predict_energy(a_test, return_error=True))
        
        F = krr.predict_force(a_test, with_error=True).reshape((-1,3))
        Fnum =  finite_diff(krr, a_test)
        print(F)
        print('')
        print(Fnum)
        print('')
        print((F - Fnum)/F)


    def test3():
        traj = read('graphene_data/all_every10th.traj', index='0::5')
        a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = a_train[0].get_number_of_atoms()

        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        a_test = traj[99]
        
        def createData2(a0, r):
            a = a0.copy()
            positions = a.get_positions()
            positions[0,0] += r
            a.set_positions(positions)
            return a

        Ntest = 100
        r_test = np.linspace(-0.3, 0.3, Ntest)
        E_test = np.zeros(Ntest)
        err_test = np.zeros(Ntest)
        
        F_test = np.zeros(Ntest)
        F_num = np.zeros(Ntest)
        val_up = np.zeros(Ntest)
        val_down = np.zeros(Ntest)
        for i, r in enumerate(r_test):
            ai = createData2(a_test, r)
            E, err, _ = krr.predict_energy(ai, return_error=True)
            E_test[i] = E
            err_test[i] = err
            
            F_test[i] = krr.predict_force(ai, with_error=True)[0]
            #F_num[i] = finite_diff(krr, ai)[0,0]
            F_num[i], val_up[i], val_down[i]  = finite_diff(krr, ai, with_ud=True)
            
            
        plt.figure()
        plt.plot(r_test, E_test, label='model')
        plt.plot(r_test, E_test-err_test, label='val')
        plt.legend()

        plt.figure()
        val = E_test-err_test
        plt.plot(r_test, val_up-val, label='val up')
        plt.plot(r_test, val_down-val, label='val down')
        plt.legend()
        
        plt.figure()
        plt.plot(r_test, F_test, label='model')
        plt.plot(r_test, F_num, 'k:', label='num')
        plt.legend()

        
    test3()
    
    plt.show()
