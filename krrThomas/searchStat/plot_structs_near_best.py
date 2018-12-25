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

def plot_near_best(traj, MLmodel):
    Ndata = len(traj)
    # Sort traj after energy
    E = np.array([a.get_potential_energy() for a in traj])
    index_best = np.argmin(E)
    a_best = traj[index_best]

    f_traj = MLmodel.featureCalculator.get_featureMat(traj)
    f_best = MLmodel.featureCalculator.get_feature(a_best)
    d = cdist(f_best.reshape((1,len(f_best))), f_traj, metric='euclidean')
    index_closest = np.argsort(d[0])[:5]
    print('d:\n', d[0][index_closest])
    print('E:\n', E[index_closest])
    traj_nearby = [traj[i] for i in index_closest]
    return traj_nearby
    
if __name__ == '__main__':

    n = 2
    i = 0
    traj_init = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_initTrain.traj'.format(n,i,i), index=':')
    traj_sp = read('/home/mkb/DFT/gpLEA/anatase/step/sanity_check/test_new_calc/runs{}/run{}/global{}_spTrain.traj'.format(n,i,i), index=':')
    traj = traj_init + traj_sp

    #ref = read('/home/mkb/DFTB/TiO_2layer/ref/Ti13O26_GM_done.traj', index='0')
    
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

    traj_nearby = plot_near_best(traj, krr)
    view(traj_nearby)
