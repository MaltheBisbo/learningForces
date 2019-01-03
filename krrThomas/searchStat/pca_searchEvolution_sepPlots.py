import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_ase import krr_class

from ase.io import read

from sklearn.decomposition import PCA




def plot_pcaEvolution(structures, featureCalculator, structures_ref=None, zoom=False, figname='pca.pdf'):
    f = []
    
    for structures_i in structures:
        f_i = featureCalculator.get_featureMat(structures_i)
        f.append(f_i)

    f_all = f[0]
    for i in range(1,len(f)):
        f_all = np.r_[f_all,f[i]]
    if structures_ref is not None:
        f_ref = featureCalculator.get_featureMat(structures_ref)
        f_all = np.r_[f_all, f_ref]
    
    pca = PCA(n_components=2)
    pca.fit(f_all)
    components = pca.components_
    variance = pca.explained_variance_ratio_
    cum_variance = np.cumsum(variance)
    print(variance)
    print(cum_variance)
    
    if structures_ref is not None:
        #f_ref = featureCalculator.get_featureMat(structures_ref)
        f2d_ref = pca.transform(f_ref)

    # Determine axis
    f2d_all = pca.transform(f_all)
    
    f2d1_all_min = np.min(f2d_all[:,0])
    f2d2_all_min = np.min(f2d_all[:,1])
    
    f2d1_all_max = np.max(f2d_all[:,0])
    f2d2_all_max = np.max(f2d_all[:,1])

    d_f2d1_all = f2d1_all_max - f2d1_all_min
    d_f2d2_all = f2d2_all_max - f2d2_all_min

    # Determine axis for zoom plots
    E_all = []
    for traj in structures:
        E_all += [a.get_potential_energy() for a in traj]
    try:
        len(structures_ref)
        E_all = E_all + [a.get_potential_energy() for a in structures_ref]
    except:
        E_all = E_all + structures_ref.get_potential_energy()
    E_all = np.array(E_all)
    if zoom:
        frac_included = 0.98
    else:
        frac_included = 1
    Npoints = int(frac_included*len(E_all))
    index_Eordered_all = np.argsort(E_all)
    search_iter_zoom = index_Eordered_all[:Npoints]
    E_all_zoom = E_all[search_iter_zoom]
    E_all_zoom_min = np.min(E_all_zoom)
    E_all_zoom_max = np.max(E_all_zoom)
    f2d_all_zoom = pca.transform(f_all[search_iter_zoom])
    
    f2d1_all_zoom_min = np.min(f2d_all_zoom[:,0])
    f2d2_all_zoom_min = np.min(f2d_all_zoom[:,1])
    
    f2d1_all_zoom_max = np.max(f2d_all_zoom[:,0])
    f2d2_all_zoom_max = np.max(f2d_all_zoom[:,1])

    d_f2d1_all_zoom = f2d1_all_zoom_max - f2d1_all_zoom_min
    d_f2d2_all_zoom = f2d2_all_zoom_max - f2d2_all_zoom_min 

    
    for i, structures_i in enumerate(structures):
        E = np.array([a.get_potential_energy() for a in structures[i]])
        f2d = pca.transform(f[i])
        Niter = len(f[i])

        search_iter = np.arange(Niter)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Colored by search iteration, N={}'.format(Niter))
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        #ax.set_xlim(f2d1_all_min-0.1*d_f2d1_all, f2d1_all_max+0.1*d_f2d1_all)
        #ax.set_ylim(f2d2_all_min-0.1*d_f2d2_all, f2d2_all_max+0.1*d_f2d2_all)
        ax.set_xlim(f2d1_all_zoom_min-0.1*d_f2d1_all_zoom, f2d1_all_zoom_max+0.1*d_f2d1_all_zoom)
        ax.set_ylim(f2d2_all_zoom_min-0.1*d_f2d2_all_zoom, f2d2_all_zoom_max+0.1*d_f2d2_all_zoom)
        cm = plt.cm.get_cmap('RdYlBu_r')
        sc = ax.scatter(f2d[:,0], f2d[:,1], c=search_iter, alpha=0.7, vmin=0, vmax=Niter, cmap=cm)
        if structures_ref is not None:
            ax.scatter(f2d_ref[:,0], f2d_ref[:,1], marker='x', c='k')
        cbar = plt.colorbar(sc)
        cbar.set_label('search iteration')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Colored by energy')
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        #ax.set_xlim(f2d1_all_min-0.1*d_f2d1_all, f2d1_all_max+0.1*d_f2d1_all)
        #ax.set_ylim(f2d2_all_min-0.1*d_f2d2_all, f2d2_all_max+0.1*d_f2d2_all)
        ax.set_xlim(f2d1_all_zoom_min-0.1*d_f2d1_all_zoom, f2d1_all_zoom_max+0.1*d_f2d1_all_zoom)
        ax.set_ylim(f2d2_all_zoom_min-0.1*d_f2d2_all_zoom, f2d2_all_zoom_max+0.1*d_f2d2_all_zoom)
        cm = plt.cm.get_cmap('RdYlBu_r')
        #sc = ax.scatter(f2d[:,0], f2d[:,1], c=E, alpha=0.7, vmin=np.min(E), vmax=np.max(E), cmap=cm)
        sc = ax.scatter(f2d[:,0], f2d[:,1], c=E, alpha=0.7, vmin=np.min(E_all_zoom), vmax=np.max(E_all_zoom), cmap=cm)
        if structures_ref is not None:
            ax.scatter(f2d_ref[:,0], f2d_ref[:,1], marker='x', c='k')
        cbar = plt.colorbar(sc)
        cbar.set_label('Energy')

    #plt.savefig(figname)
    plt.show()

if __name__ == '__main__':
    
    structures = []
    structures.append(read('/home/mkb/DFT/C24/all_runs/runs22/run0/global0_spTrain.traj', index=':'))
    n_list = [0,1]
    for n in n_list:
        structures_temp = read('/home/mkb/DFT/gpLEA/C24/runs2/run{}/global{}_spTrain.traj'.format(n,n), index=':')
        structures.append(structures_temp)
    n_list = [2,3]
    for n in n_list:
        structures_temp = read('/home/mkb/DFT/gpLEA/C24/runs3/run{}/global{}_spTrain.traj'.format(n,n), index=':')
        structures.append(structures_temp)
    n_list = [0,4]
    for n in n_list:
        structures_temp = read('/home/mkb/DFT/gpLEA/C24/runs4/run{}/global{}_spTrain.traj'.format(n,n), index=':')
        structures.append(structures_temp)
    
    structures_ref = read('/home/mkb/DFT/C24/Fullerene/full_4lowest_new.traj', index=':')

    """
    n_list =[7,8,9]
    structures = []
    for n in n_list:
        structures_temp = read('/home/mkb/DFT/gpLEA/Sn3O3/runs_stat/runs{}/run1/global1_spTrain.traj'.format(n), index=':')
        structures.append(structures_temp)
    structures_ref = None
    """

    a0 = structures[0][0]

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

    plot_pcaEvolution(structures, featureCalculator, structures_ref=structures_ref, zoom=True)
