import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS, BFGSLineSearch
from ase.constraints import FixedPlane
from ase.io import Trajectory
from ase.ga.relax_attaches import VariansBreak

def create2Dstructure(Natoms, rmin=0.9, rmax=1.8):
    dim = 3
    boxsize = 1.5 * np.sqrt(Natoms)

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

def relax_VarianceBreak(structure, calc, label, niter_max=10):
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
    #if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
    #    return structure

    traj = Trajectory(label+'.traj','a', structure)
    for niter in range(niter_max):
        if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
            return structure
        dyn = BFGSLineSearch(structure,
                             logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 500)
        niter += 1

    return structure

def N_relaxed(Nstructs, Natoms, calculator, label):

    for n in range(Nstructs):
        print('progress: {}/{}'.format(n, Nstructs))
        traj_label = label+'{}'.format(n)
        # Generate new structure
        struct_new = create2Dstructure(Natoms)

        # Constrain structure to xy-plane
        plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(struct_new))]
        struct_new.set_constraint(plane)

        struct = relax_VarianceBreak(struct_new, calculator, traj_label)

        # Relax
        # struct_new.set_calculator(calculator)
        # dyn = BFGS(struct_new, trajectory=label+'{}.traj'.format(n))
        # dyn.run(fmax=0.1)            

       
if __name__ == '__main__':
    from custom_calculators import doubleLJ_calculator
    from ase.io import read
    
    calc = doubleLJ_calculator(noZ=True)
    
    label = 'dLJ19data2/'
    N_relaxed(Nstructs=100,
              Natoms=19,
              calculator=calc,
              label=label)
    

    """
    a = read('dLJ19data/29.traj', index='-1')
    view(a)
    a.set_calculator(calc)
    print(a.get_potential_energy())
    print(a.get_forces())

    # Constrain structure to xy-plane
    plane = [FixedPlane(x, (0, 0, 1)) for x in range(len(a))]
    a.set_constraint(plane)
    
    # Relax
    dyn = BFGS(a, trajectory='test.traj')
    dyn.run(fmax=0.1)
    """

    
