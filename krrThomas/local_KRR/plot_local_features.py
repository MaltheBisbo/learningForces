import numpy as np
import matplotlib.pyplot as plt
from featureCalculators_local.angular_fingerprintFeature_local_cy import Angular_Fingerprint_local

from ase.io import read
from ase import Atoms
from ase.visualize import view


def plotLocal_pos(a, i):
    pos = a.get_positions()
    pos = pos[:, :2]  # remove z-component
    plt.plot(pos[:i, 0], pos[:i, 1], 'ko', ms=20)
    plt.plot(pos[i, 0], pos[i, 1], 'ro', ms=20)
    plt.plot(pos[i+1:, 0], pos[i+1:, 1], 'ko', ms=20)

def plotLocal_all(a):
    feature = featureCalculator.get_feature(a)
    Nelements = feature.shape[1]
    Natoms = a.get_number_of_atoms()
    d = 2
    space = 2
    xsize = 2*(d + space) + space
    ysize = Natoms*(d + space) + space
    #plt.figure(figsize=(ysize, xsize))
    plt.rcParams["figure.figsize"] = (xsize, ysize)
    plt.subplots_adjust(left=space/xsize, right=(xsize-space)/xsize, wspace=space/xsize,
                        bottom=space/ysize, top=(ysize-space)/ysize, hspace=space/ysize)
    for i in range(Natoms):
        plt.subplot(Natoms,2,2*i+1, aspect='equal')
        plotLocal_pos(a, i)
        
        plt.subplot(Natoms,2,2*i+2)
        plt.plot(np.arange(Nelements), feature[i])


if __name__ == '__main__':

    dim = 3
    x = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 1.5, 1, 0])
    positions = x.reshape((-1,dim))
    a = Atoms('H4',
              positions=positions,
              cell=[4,2,1],
              pbc=[0, 0, 0])

    #view(a)

    Rc1 = 5
    binwidth1 = 0.05
    sigma1 = 0.2
 
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
 
    gamma = 1
    eta = 30
    use_angular = False
 
    featureCalculator = Angular_Fingerprint_local(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
    
    plotLocal_all(a)
    plt.savefig('../results/local_features/H4.pdf')
    plt.show()
