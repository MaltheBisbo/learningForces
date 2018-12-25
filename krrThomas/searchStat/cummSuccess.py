import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write

def calc_cummSuccess(success_indices, Nruns, run_length):
    cummSuccess = np.zeros(run_length)
    for index in success_indices:
        cummSuccess[index:] += 1
    cummSuccess /= Nruns
    return cummSuccess

def cummSuccess(a_gm, runs_path, featureCalculator, dmax=0.1, N=100, traj_name=None):
    f_gm = featureCalculator.get_feature(a_gm)
    Nruns = 0
    n_success_list = []
    a_success_list = []
    iter_max = 0
    for i in range(N):
        try:
            traj = read(runs_path + 'run{}/global{}_spTrain.traj'.format(i,i), index=':')
            Nruns += 1
            if len(traj) > iter_max:
                iter_max = len(traj)
        except:
            break
        fMat = featureCalculator.get_featureMat(traj)
        for i, (f, a) in enumerate(zip(fMat, traj)):
            d = euclidean(f, f_gm)
            if d < dmax:
                n_success_list.append(i)
                a_success_list.append(a)
                break

    if traj_name is not None:
        write(traj_name, a_success_list)

    searchIndex = np.arange(iter_max)
    cummSuccess = calc_cummSuccess(n_success_list, Nruns, iter_max)
    fig, ax = plt.subplots()
    ax.plot(searchIndex, cummSuccess)
    plt.show()

    
        
        
        
        
        
        
