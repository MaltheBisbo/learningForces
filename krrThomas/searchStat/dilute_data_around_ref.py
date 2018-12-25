import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from ase.io import read, write

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_errorForce import krr_class

def check_dist(traj, ref, MLmodel):
    f_traj = MLmodel.featureCalculator.get_featureMat(traj)
    f_ref = MLmodel.featureCalculator.get_feature(ref)
    d = cdist(f_ref.reshape((1,len(f_ref))), f_traj, metric='euclidean')[0] 
    print('dist to closest train:\n', np.sort(d)[:5])

def dilute(traj, a_dilute, MLmodel, d_dilute):
    f_traj = MLmodel.featureCalculator.get_featureMat(traj)
    f_dilute = MLmodel.featureCalculator.get_feature(a_dilute)
    d = cdist(f_dilute.reshape((1,len(f_dilute))), f_traj, metric='euclidean') 
    filt = d[0] > d_dilute
    traj_diluted = [traj[i] for i in range(len(traj)) if filt[i]]

    # test
    f_traj_diluted = f_traj[filt]
    d_diluted = cdist(f_dilute.reshape((1,len(f_dilute))), f_traj_diluted, metric='euclidean')[0]
    print('dist to closest:\n', np.sort(d_diluted)[:5])
    return traj_diluted
    
def dilute_data_around_ref(traj, MLmodel, sigma=30, reg=1e-5, ref=None, interval=10, d_dilute=[0, 0.2, 0.4], plot_fitness=False):
    Nall = len(traj)
    Eref_pred = []
    error_ref_pred = []
    Ncur = []

    if ref is None:
        traj_current = traj
        # use best structure found as first dilute center
        Etraj = np.array([a.get_potential_energy() for a in traj_current])
        index_best = np.argmin(Etraj)
        ref = traj[index_best]
        
    for d in d_dilute:
        print(d, 'start')
        traj_diluted = dilute(traj_current, ref, MLmodel, d_dilute=d)

        traj_train = traj_diluted + [ref]
        GSkwargs = {'reg': [reg], 'sigma': [sigma]}
        FVU, params = MLmodel.train(atoms_list=traj_train,
                                    add_new_data=False,
                                    k=3,
                                    **GSkwargs)
        check_dist(traj_train, ref, MLmodel)
        Ncur.append(len(traj_train))

        write('diluted_d{}.traj'.format(d), traj_train)
        
        # predicted energy of reference structure
        Eref_pred_i, error_ref_pred_i, _ = MLmodel.predict_energy(ref, return_error=True)
        print(Eref_pred_i)
        Eref_pred.append(Eref_pred_i)
        error_ref_pred.append(error_ref_pred_i)

    n = Nall - np.array(Ncur)
    print(Nall)
    print(Ncur)
    print(n)
        
    Eref_pred = np.array(Eref_pred)
    error_ref_pred = np.array(error_ref_pred)
        
    Eref_true = ref.get_potential_energy()
    Eref_true_array = Eref_true*np.ones(len(n))
    
    Epred_last = traj[-1].info['key_value_pairs']['predictedEnergy']
    errpred_last = traj[-1].info['key_value_pairs']['predictedError']
    fitness_last = traj[-1].info['key_value_pairs']['fitness']
    kappa = (Epred_last - fitness_last)/errpred_last
    
    fitness_ref_pred = Eref_pred - kappa*error_ref_pred

    plt.figure()
    plt.plot(d_dilute, Eref_pred, label='Eref_pred')
    if plot_fitness:
        plt.plot(d_dilute, fitness_ref_pred, label='fitness_ref_pred')
    plt.plot(d_dilute, Eref_true_array, label='Eref_true')
    #plt.scatter(d, Eref_pred, color='k')
    plt.xlabel('# data removed')
    plt.ylabel('Energy')
    plt.legend()



    
    
if __name__ == '__main__':
    """
    n = 2
    i = 0
    traj_init = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    #ref = read('/home/mkb/DFT/gpLEA/anatase/step/bestfound_2u.traj', index='1')
    """
    n = 18
    i = 18
    traj_init = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    ref = read('/home/mkb/DFTB/TiO_2layer/ref/Ti13O26_GM_done.traj', index='0')

-    ### Set up feature ###

    # Template structure
    a = traj[0]
    
    # Radial part
    Rc1 = 6
    binwidth1 = 0.2
    sigma1 = 0.2
    
    # Angular part
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    gamma = 2
    
    # Radial/angular weighting
    eta = 20
    use_angular = True
    
    # Initialize feature
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    ### Set up KRR-model ###
    comparator = gaussComparator(featureCalculator=featureCalculator, max_looks_like_dist=0.2)
    delta_function = deltaFunc(atoms=a, rcut=6)
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator,
                    delta_function=delta_function,
                    bias_std_add=0)

    def new(traj, krr, d_dilute):
        # use best structure found as first dilute center
        E = np.array([a.get_potential_energy() for a in traj])
        index_best = np.argmin(E)
        ref = traj[index_best]
        Eref = ref.get_potential_energy()
        traj_base = dilute(traj, ref, krr, d_dilute=d_dilute)
        traj_train = traj_base + [ref]
        print('Ntrain:', len(traj_train))
        GSkwargs = {'reg': [1e-5], 'sigma': [30]}
        FVU, params = krr.train(atoms_list=traj_train,
                                add_new_data=False,
                                k=3,
                                **GSkwargs)
        Eref_pred, error_ref_pred, _ = krr.predict_energy(ref, return_error=True)
        print(Eref_pred)
        return Eref_pred, error_ref_pred, Eref
    """
    Epred_list = []
    err_pred_list = []
    Eref_list = []
    d_list = [0,3,10]
    for d in d_list:
        Epred, err_pred, Eref = new(traj, krr, d_dilute=d)
        Epred_list.append(Epred)
        err_pred_list.append(err_pred)
        Eref_list.append(Eref)

    plt.figure()
    plt.plot(d_list, Epred_list)
    plt.plot(d_list, Eref_list)
    plt.show()
    """
        
        
    if False:
        # use best structure found as first dilute center
        E = np.array([a.get_potential_energy() for a in traj])
        index_best = np.argmin(E)
        ref = traj[index_best]
        Eref = ref.get_potential_energy()
        traj_base = dilute(traj, ref, krr, d_dilute=0)
        print('Ntrain:', len(traj_base))
        traj_train = traj_base + [ref]
        GSkwargs = {'reg': [1e-5], 'sigma': [30]}
        FVU, params = krr.train(atoms_list=traj_train,
                                k=3,
                                **GSkwargs)
        Eref_pred, error_ref_pred, _ = krr.predict_energy(ref, return_error=True)
        print('Eref:', Eref)
        print('Eref_pred:', Eref_pred)
        print('Eref_pred:', krr.predict_energy(ref))
    if True:
        dilute_data_around_ref(traj, krr, sigma=30, reg=1e-5, d_dilute=[0, 3, 10])
        plt.savefig('test.pdf')
        plt.show()
        
