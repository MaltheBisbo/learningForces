import numpy as np
import matplotlib.pyplot as plt

def plotLC(path_core, title, fignum):
    Natoms = 7
    Nruns = 79
    Npoints = 30
    Ndata_array = None
    E = np.zeros((Nruns, Npoints))
    F = np.zeros((Nruns, Npoints, 2*Natoms))
    for i in range(Nruns):
        data = np.loadtxt(path_core + str(i) + '.dat', delimiter='\t')
        if Ndata_array is None:
            Ndata_array = data[:,0]
        E[i] = data[:,1]
        F[i] = data[:, 2:]
            
    meanFVU_energy = np.mean(E, axis=0)
    meanFVU_force = np.mean(F, axis=0)
    
    plt.figure(fignum)
    plt.title('Learning curve ' + title + ' (Energy)')
    plt.loglog(Ndata_array, meanFVU_energy)
    plt.xlabel('# training data')
    plt.ylabel('Fraction of variance unexplained')
    plt.figure(fignum+1)
    plt.title('Learning curve ' + title + ' (Force)')
    plt.loglog(Ndata_array, meanFVU_force)
    plt.xlabel('# training data')
    plt.ylabel('Fraction of variance unexplained')

if __name__ == '__main__':
    path_core1 = 'dataN7_finger_random3/LC_finger_random'
    path_core2 = 'dataN7_finger_search3/LC_finger_search_perm'
    path_core3 = 'dataN7_finger_search_perm3/LC_finger_search_perm'
    
    plotLC(path_core1, 'random structures', 1)
    plotLC(path_core2, 'ordered search structures', 3)
    plotLC(path_core3, 'unordered search structures', 5)
    plt.show()
