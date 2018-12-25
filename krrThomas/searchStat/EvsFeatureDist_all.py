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

def plot_EvsDist(traj, MLmodel):
    Ndata = len(traj)
    E = np.array([a.get_potential_energy() for a in traj]).reshape((-1,1))
    f_traj = MLmodel.featureCalculator.get_featureMat(traj)
    d = cdist(f_traj, f_traj, metric='euclidean') 

    # Test
    triu_indices = np.triu_indices(Ndata,k=1)
    d_triu = d[triu_indices]
    dE = cdist(E, E, metric='euclidean')
    dE_triu = dE[triu_indices]

    try:
        print('max grad:', np.max(dE_triu/d_triu))
    except Exception as err:
        print(err)
    plt.figure()
    plt.xlabel('distance')
    plt.ylabel('Energy difference')
    plt.scatter(d_triu, dE_triu, alpha=0.05)

if __name__ == '__main__':

    """
    n = 2
    i = 0
    traj_init = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    ref = read('/home/mkb/DFT/gpLEA/anatase/step/bestfound_2u.traj', index='1')
    """

    n = 9
    i = 0
    traj = read('/home/mkb/DFTB/TiO_3layer/dualGPR_search/runs{}/run{}/global{}_fineTrain.traj'.format(n,i,i), index=':')
    #traj = read('/home/mkb/DFTB/TiO_3layer/dualGPR_search/runs{}/run{}/global{}_priorTrain.traj'.format(n,i,i), index=':')
    #traj_sp = read('/home/mkb/DFTB/TiO_3layer/dualGPR_search/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    #traj = traj_init + traj_sp

    """
    n = 2
    i = 13
    traj_init = read('/home/mkb/DFTB/TiO_3layer/dualGPR_search/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFTB/TiO_3layer/dualGPR_search/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp
    """

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

    plot_EvsDist(traj, krr)
    plt.savefig('EvsDist_prior.pdf')
    
    Esorted = np.sort(np.sort([a.get_potential_energy() for a in traj]))
    n = np.arange(len(traj))
    plt.figure()
    plt.plot(n,Esorted)
    plt.show()
