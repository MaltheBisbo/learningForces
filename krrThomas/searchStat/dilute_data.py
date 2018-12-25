import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_errorForce import krr_class

def dilute_data(traj, featureCalculator, d_dilute):
    Ndata = len(traj)
    # Sort traj after energy
    E = np.array([a.get_potential_energy() for a in traj])
    sort_indices = np.argsort(E)
    traj_sort = [traj[i] for i in sort_indices]
    
    f_traj = featureCalculator.get_featureMat(traj_sort)
    d = cdist(f_traj, f_traj, metric='euclidean') 

    indices_use = np.arange(Ndata)
    for index1 in range(Ndata):
        if index1 >= len(indices_use):
            break
        else:
            i = indices_use[index1]
        Ndata_remaining = len(indices_use)
        for index2 in reversed(range(index1+1,Ndata_remaining)):
            j = indices_use[index2]
            if d[i,j] < d_dilute:
                indices_use = np.delete(indices_use, index2)
                #del indices_use[j]

    traj_diluted = [traj_sort[i] for i in indices_use]

    return traj_diluted

def get_distances(traj, featureCalculator):
    Ndata = len(traj)
    f_traj = featureCalculator.get_featureMat(traj)
    d = cdist(f_traj, f_traj, metric='euclidean')
    triu_indices = np.triu_indices(Ndata,k=1)
    d_unique = d[triu_indices]
    return d_unique

def get_distance2ref(traj, ref, featureCalculator):
    Ndata = len(traj)
    f_traj = featureCalculator.get_featureMat(traj)
    f_ref = featureCalculator.get_feature(ref)
    d = cdist(f_ref.reshape((1,len(f_ref))), f_traj, metric='euclidean')[0]
    return d


if __name__ == '__main__':

    n = 2
    i = 0
    traj_init = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':50')
    traj = traj_init + traj_sp

    ref = read('/home/mkb/DFT/gpLEA/anatase/step/bestfound_2u.traj', index='0')

    """
    n = 18
    i = 18
    traj_init = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    ref = read('/home/mkb/DFTB/TiO_2layer/ref/Ti13O26_GM_done.traj', index='0')
    """

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
    
    traj_diluted = dilute_data(traj, krr.featureCalculator, d_dilute=-1, use_initial_order=True)
    N = len(traj)
    N_diluted = len(traj_diluted)
    E = np.array([a.get_potential_energy() for a in traj])
    E_diluted = np.array([a.get_potential_energy() for a in traj_diluted])
    plt.figure()
    plt.plot(np.arange(N), E)
    plt.plot(np.arange(N_diluted), E_diluted, 'k:')
    plt.show()

    """
    d = get_distances(traj_diluted, krr)

    print('N_all:', len(traj))
    print('N_diluted:', len(traj_diluted))
    print(np.sort(d)[:50])
    """

    
    """
    d = get_distance2ref(traj, ref, krr)
    d_sorted = np.sort(d)
    print(d_sorted[:50])
    
    plt.figure()
    plt.semilogy(np.arange(len(d_sorted)), d_sorted)
    plt.title('distance to best in run')
    plt.xlabel('# sp calculations')
    plt.ylabel('distance')
    
    plt.savefig('dist2bestInRun_runs2run0.pdf', transparent=True)
    plt.show()
    """
