import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from ase.io import read, write

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_errorForce import krr_class

def dilute(traj, a_dilute, MLmodel, d_dilute):
    f_traj = MLmodel.featureCalculator.get_featureMat(traj)
    f_dilute = MLmodel.featureCalculator.get_feature(a_dilute)
    #d = cdist(f_dilute, f_traj, metric='euclidean')
    d = cdist(f_dilute.reshape((1,len(f_dilute))), f_traj, metric='euclidean') 
    filt = d[0] > d_dilute
    traj_diluted = [traj[i] for i in range(len(traj)) if filt[i]]

    index_next = np.argmin(d[0,filt])
    a_next = traj_diluted[index_next]
    return traj_diluted, a_next
    
def predict_best_diluted(traj, ref, MLmodel, sigma, interval=10, d_dilute=0.2):
    Nall = len(traj)
    Eref_pred = []
    error_ref_pred = []
    Ncur = []
    diluted = []

    GSkwargs = {'reg': [1e-5], 'sigma': [sigma]}
    FVU, params = MLmodel.train(atoms_list=traj,
                                k=3,
                                **GSkwargs)
    Ncur.append(len(traj))
    
    # predicted energy of reference structure
    Eref_pred_i, error_ref_pred_i, _ = MLmodel.predict_energy(ref, return_error=True)
    Eref_pred.append(Eref_pred_i)
    error_ref_pred.append(error_ref_pred_i)
    
    traj_current = traj
    # use best structure found as first dilute center
    Etraj = np.array([a.get_potential_energy() for a in traj_current])
    index_best = np.argmin(Etraj)
    a2dilute = traj[index_best]
    for i in range(5):
        print(i, 'start')
        
        if i==0:
            traj_current, a_next = dilute(traj_current, a2dilute, MLmodel, d_dilute=d_dilute)
        else:
            traj_current, a_next = dilute(traj_current[:-i], a2dilute, MLmodel, d_dilute=d_dilute)
        print(i, 'diluting done')
        diluted.append(a2dilute)

        traj_train = traj_current + diluted
        GSkwargs = {'reg': [1e-5], 'sigma': [sigma]}
        FVU, params = MLmodel.train(atoms_list=traj_train,
                                    k=3,
                                    **GSkwargs)
        Ncur.append(len(traj_train))
        
        # predicted energy of reference structure
        Eref_pred_i, error_ref_pred_i, _ = MLmodel.predict_energy(ref, return_error=True)
        Eref_pred.append(Eref_pred_i)
        error_ref_pred.append(error_ref_pred_i)

        a2dilute = a_next

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
    plt.plot(n, Eref_pred, label='Eref_pred')
    plt.plot(n, fitness_ref_pred, label='fitness_ref_pred')
    plt.plot(n, Eref_true_array, label='Eref_true')
    plt.scatter(n, Eref_pred, color='k')
    plt.ylim(Eref_true-5, Eref_true+10)
    plt.xlabel('# data removed')
    plt.ylabel('Energy')
    plt.legend()


    
if __name__ == '__main__':

    n = 18
    i = 18
    traj_init = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    ref = read('/home/mkb/DFTB/TiO_2layer/ref/Ti13O26_GM_done.traj', index='0')
    
    ### Set up feature ###

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

    predict_best_diluted(traj, ref, krr, sigma=30)
    plt.show()
