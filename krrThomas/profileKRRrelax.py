import numpy as np

from gaussComparator import gaussComparator
#from angular_fingerprintFeature import Angular_Fingerprint
from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
from krr_ase import krr_class
from custom_calculators import krr_calculator

from ase.io import read, write, Trajectory
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.ga.relax_attaches import VariansBreak


def relax_VarianceBreak(structure, calc, label):
    '''
    Relax a structure and saves the trajectory based in the index i

    Parameters
    ----------
    structure : ase Atoms object to be relaxed

    Returns
    -------
    '''

    # Set calculator 
    structure.set_calculator(calc)

    # loop a number of times to capture if minimization stops with high force
    # due to the VariansBreak calls
    forcemax = 0.1
    niter = 0

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    traj = Trajectory(label+'.traj','a', structure)
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < 5:
        dyn = BFGS(structure, logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 500)
        niter += 1




atoms_relax = read('grendel/DFTBrelax1/global_ML005.traj', index='0')
atoms_train1 = read('grendel/DFTBrelax1/global_initTrain.traj', index=':')
atoms_train2 = read('grendel/DFTBrelax1/global_spTrain.traj', index=':')
atoms_train = atoms_train1 + atoms_train2
a0 = atoms_train[0]

print('# train1', len(atoms_train1))
print('# train2', len(atoms_train2))

Rc1 = 5
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 1
eta = 30
use_angular = True

featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

# Set up KRR-model
comparator = gaussComparator()
krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, bias_fraction=0.7, bias_std_add=1)

GSkwargs = {'reg': [1e-5], 'sigma': np.logspace(0,2,10)}
MAE, params = krr.train(atoms_train, add_new_data=False, k=10, **GSkwargs)
print('MAE=', MAE)

calc = krr_calculator(krr)
plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(atoms_relax))]
atoms_relax.set_constraint(plane)

label = 'profile'
relax_VarianceBreak(atoms_relax, calc, label)

#atoms_relax.set_calculator(calc)
#dyn = BFGS(atoms_relax, trajectory='profile.traj')
#dyn.run(fmax=0.1)
