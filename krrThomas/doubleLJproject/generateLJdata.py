import numpy as np
from ase import Atoms
from ase.visualize import view
from custom_calculators import doubleLJ_calculator

from ase.io import read, write, Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.ga.utilities import closest_distances_generator
from ase.optimize import BFGS, BFGSLineSearch
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixedPlane
from ase.ga.relax_attaches import VariansBreak

def relax_VarianceBreak(structure, calc, label, niter_max=10, steps=500):
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

    traj = Trajectory(label+'.traj','a', structure)
    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        traj.write(structure)
        return structure
    
    for niter in range(niter_max):
        if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
            return structure
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax=forcemax, steps=steps)
        niter += 1

    return structure

def createInitalStructure2d_LJ(Natoms):
    dim = 3
    boxsize = 2 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 1.5

    def validPosition(X, xnew):
        Natoms = int(len(X)/dim)  # Current number of atoms
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[dim*i:dim*(i+1)])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    X = np.zeros(dim*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.r_[np.random.rand(dim-1) * boxsize, boxsize/2]
            if validPosition(X[:dim*i], xnew):
                X[dim*i:dim*(i+1)] = xnew
                break
    X = X.reshape((-1, 3))

    atomtypes = str(Natoms) + 'He'
    pbc = [0,0,0]
    cell = [boxsize]*3
    a = Atoms(atomtypes,
              positions=X,
              pbc=pbc,
              cell=cell)
    return a

Natoms = 19
Ndata = 1000
calc = doubleLJ_calculator()

for i_relax in range(Ndata):
    print(i_relax)
    label = 'LJdata/LJ{}/relax{}'.format(Natoms, i_relax)
    a = createInitalStructure2d_LJ(Natoms)
    plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
    a.set_constraint(plane)
    relax_VarianceBreak(a, calc, label)
