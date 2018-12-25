import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_ase import krr_class

from ase.io import read

from sklearn.decomposition import PCA




def plot_pcaEvolution([structures], featureCalculator, structures_ref=None):
    E = np.array([a.get_potential_energy() for a in structures])
    Niter = len(structures)

    f = featureCalculator.get_featureMat(structures)
    #f_train[:,:25] = 0
    
    pca = PCA(n_components=6)
    pca.fit(f)
    components = pca.components_
    variance = pca.explained_variance_ratio_
    cum_variance = np.cumsum(variance)
    print(variance)
    print(cum_variance)
    
    f2d = pca.transform(f)

    if structures_ref is not None:
        f_ref = featureCalculator.get_featureMat(structures_ref)
        f2d_ref = pca.transform(f_ref)
    
    
    fig = plt.figure()
    #for i in range(1, len(structures)+1):
    color_weights = np.arange(Niter)
    ax = fig.add_subplot(111)
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    cm = plt.cm.get_cmap('RdYlBu_r')
    sc = ax.scatter(f2d[:,0], f2d[:,1], c=color_weights, alpha=0.7, vmin=0, vmax=Niter, cmap=cm)
    if structures_ref is not None:
        ax.scatter(f2d_ref[:,0], f2d_ref[:,1], marker='x', c='k')
    cbar = plt.colorbar(sc)
    cbar.set_label('search iteration')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    cm = plt.cm.get_cmap('RdYlBu_r')
    sc = ax.scatter(f2d[:,0], f2d[:,1], c=E, alpha=0.7, vmin=np.min(E), vmax=np.max(E), cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Energy')
    
    
    color_weights = np.arange(Niter)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_zlabel('Energy')
    cm = plt.cm.get_cmap('RdYlBu_r')
    sc = ax.scatter(f2d[:,0], f2d[:,1], E, c=color_weights, alpha=0.7, vmin=0, vmax=Niter, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label('search iteration')
    
    plt.show()

if __name__ == '__main__':
    structures = read('/home/mkb/DFT/C24/all_runs/runs21/run1/global1_spTrain.traj', index=':')
    #structures_ref = read('/home/mkb/DFT/C24/Fullerene/full_4lowest_new.traj', index=':')

    """
    n_list =[7,8,9]
    structures = []
    for n in n_list:
        structures_temp = read('/home/mkb/DFT/gpLEA/Sn3O3/runs_stat/runs8/run0/global0_spTrain.traj', index=':')
        structures.append(structures_temp)
    structures_ref = None
    """
    a0 = structures[0]

    """
    #C24
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 20
    use_angular = True
    """
    
    #Sn2O3
    Rc1 = 6
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    gamma = 2
    
    eta = 20
    use_angular = True
    
    
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    plot_pcaEvolution(structures, featureCalculator, structures_ref=structures_ref)
