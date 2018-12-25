import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_errorForce import krr_class

def is_stuck_test(traj, ref, MLmodel, sigma, reg=1e-5, interval=10, rel_ylim=[-5, 10], fig_name=None):
    Ebest = None
    Ebest_array = []
    Eref_pred = []
    error_ref_pred = []

    Ndata = len(traj)
    n = np.arange(0, Ndata, interval)
    for i in n:
        i_start = i
        if i + interval > Ndata:
            i_end = Ndata
        else:
            i_end = i + interval

        a_train_add = [traj[i] for i in range(i_start, i_end)]
        GSkwargs = {'reg': [reg], 'sigma': [sigma]}
        FVU, params = MLmodel.train(atoms_list=a_train_add,
                                    add_new_data=True,
                                    k=3,
                                    **GSkwargs)

        # Best structure found so fa in search
        if Ebest == None:
            Ebest = np.min(np.array([a.get_potential_energy() for a in a_train_add]))
        else:
            Ebest = np.min(np.array([a.get_potential_energy() for a in a_train_add] + [Ebest]))
        Ebest_array.append(Ebest)

        # predicted energy of reference structure
        Eref_pred_i, error_ref_pred_i, _ = MLmodel.predict_energy(ref, return_error=True)
        Eref_pred.append(Eref_pred_i)
        error_ref_pred.append(error_ref_pred_i)

    Eref_pred = np.array(Eref_pred)
    error_ref_pred = np.array(error_ref_pred)
        
    Eref_true = ref.get_potential_energy()
    Eref_true_array = Eref_true*np.ones(len(n))
    
    Nstart_fitness = 10
    n_fine = np.arange(Nstart_fitness, Ndata)
    fitness = np.array([a.info['key_value_pairs']['fitness'] for a in traj[Nstart_fitness:]])
    Epred_last = traj[-1].info['key_value_pairs']['predictedEnergy']
    errpred_last = traj[-1].info['key_value_pairs']['predictedError']
    kappa = (Epred_last - fitness[-1])/errpred_last
    print(kappa)

    fitness_ref_pred = Eref_pred - kappa*error_ref_pred
    
    plt.figure()
    plt.plot(n_fine, fitness, color='k', alpha=0.5, label='fitness best')
    plt.plot(n, Ebest_array, label='Erun_best')
    plt.plot(n, Eref_pred, label='Eref_pred')
    plt.plot(n, fitness_ref_pred, label='fitness_ref_pred')
    plt.plot(n, Eref_true_array, label='Eref_true')
    plt.ylim(Eref_true+rel_ylim[0], Eref_true+rel_ylim[1])
    plt.xlabel('singlepoint calculations')
    plt.ylabel('Energy')
    plt.legend()
    
    if fig_name is not None:
        plt.savefig(fig_name)


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

    is_stuck_test(traj, ref, krr, sigma=30)
