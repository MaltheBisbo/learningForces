import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write, Trajectory
from ase.visualize import view


from gaussComparator import gaussComparator
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class

def learningCurve(atoms_train, atoms_test, Npoints=10):
    a0 = atoms_train[0]
    Ntrain = len(atoms_train)
    Etest = np.array([a.get_potential_energy() for a in atoms_test])
    
    # Permute the training structures
    index_permut = np.random.permutation(Ntrain)
    atoms_train = [atoms_train[i] for i in index_permut]
    
    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 5
    use_angular = True
    
    featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator,
                    featureCalculator=featureCalculator)
    GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0, 3, 10)}
    
    Ntrain_array = np.logspace(np.log10(5),np.log10(500), Npoints).astype(int)
    MAE_array = []
    for Ntrain in Ntrain_array:
        atoms_train_sub = atoms_train[:Ntrain]
        MAE_train, params = krr.train(atoms_train_sub, k=5, **GSkwargs)

        Epred = [krr.predict_energy(a) for a in atoms_test]

        E_diff = Etest - np.array(Epred)
        """
        MSE = np.mean(E_diff**2)
        std = np.std(E_diff)
        FVU = MSE/std
        """
        
        MAE = np.mean(np.abs(E_diff))
        MAE_array.append(MAE)

    return MAE_array, Ntrain_array

def repeat_learningCurve(atoms_train, atoms_test, Nrepeat=3, Npoints=10):
    MAE = np.zeros(Npoints)
    for i in range(Nrepeat):
        MAE_add, N = learningCurve(atoms_train, atoms_test, Npoints)
        MAE += MAE_add
    MAE /= Nrepeat
    return MAE, N
        
if __name__ == '__main__':
    Nrepeat = 5
    atoms3 = read('LJdata/LJ3/farthestPoints.traj', index=':')
    MAE3, N = repeat_learningCurve(atoms3[:500], atoms3[500:], Nrepeat)

    atoms7 = read('LJdata/LJ7/farthestPoints.traj', index=':')
    MAE7, N = repeat_learningCurve(atoms7[:500], atoms7[500:], Nrepeat)

    atoms19 = read('LJdata/LJ19/farthestPoints.traj', index=':')
    MAE19, N = repeat_learningCurve(atoms19[:500], atoms19[500:], Nrepeat)

    plt.figure()
    plt.title('Learning curves for varying structure size')
    plt.xlabel('# training data')
    plt.ylabel('MAE')
    plt.loglog(N, MAE3, label='3 atoms')
    plt.loglog(N, MAE7, label='7 atoms')
    plt.loglog(N, MAE19, label='19 atoms')
    plt.legend()
    plt.savefig('LC_structureSize.pdf')
    plt.show()
    
