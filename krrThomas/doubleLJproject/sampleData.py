import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write, Trajectory
from ase.visualize import view

from scipy.spatial.distance import cdist

from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint

def plot_dataDistribution(atoms_list):
    pass

def randomSample(Natoms, Ndata, NperTraj=10):
    data_sample = []
    for i_relax in range(Ndata):
        label = 'LJdata/LJ{}/relax{}'.format(Natoms, i_relax)
        traj = read(label + '.traj', index=':')
        index_random = np.random.permutation(len(traj))
        index_sample = index_random[:NperTraj]
        data_sample = data_sample + [traj[i] for i in index_sample]
    return data_sample

def farthestPointsSampling(atoms_list, featureCalculator, Nsamples):
    # Calculate features + distance matrix
    fMat = featureCalculator.get_featureMat(atoms_list)
    dMat = cdist(fMat, fMat, metric='euclidean')
    Nstructs = len(atoms_list)
    index = np.arange(Nstructs)

    index_sub = [np.random.randint(Nstructs)]
    for i in range(Nsamples-1):
        index_notInSub = list(set(index) - set(index_sub))
        dMat_sub = dMat[index_sub, :][:, index_notInSub]
        index_new = dMat_sub.min(axis=0).argmax()
        index_sub.append(index_notInSub[index_new])
    atoms_sample = [atoms_list[i] for i in index_sub]
    return atoms_sample
        
if __name__ == '__main__':
    Natoms = 19
    Ndata = 1000
    Nsamples = 1000
    
    #
    label = 'LJdata/LJ{}/relax{}'.format(Natoms, 0)
    
    a0 = read(label + '.traj', index='-1')
    
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 5
    use_angular = False
    
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    # Set up KRR-model
    comparator = gaussComparator()
    

    atoms_sample = randomSample(Natoms, Ndata)
    atoms_sample_sub = farthestPointsSampling(atoms_sample, featureCalculator, Nsamples)
    write('LJdata/LJ{}/farthestPoints.traj'.format(Natoms), atoms_sample_sub)
    


