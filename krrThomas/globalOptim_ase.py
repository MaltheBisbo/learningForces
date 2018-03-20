import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import time

from startgenerator import StartGenerator
from custom_calculators import krr_calculator

from ase import Atoms
from ase.io import read, write
from ase.ga.utilities import closest_distances_generator
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixedPlane

def createInitalStructure2d(Natoms):
    dim = 3
    boxsize = 2 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 2.2

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

def createInitalStructure():
    '''
    Creates an initial structure of 24 Carbon atoms
    '''    
    number_type1 = 6  # Carbon
    number_opt1 = 24  # number of atoms
    atom_numbers = number_opt1 * [number_type1]

    cell = np.array([[24, 0, 0],
                     [0, 24, 0],
                     [0, 0, 18]])
    pbc = [False, False, False]

    template = Atoms('')
    template.set_cell(cell)
    template.set_pbc(pbc)
    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a a center position (p0)
    # and three spanning vectors
    
    a = np.array((4.5, 0., 0.))
    b = np.array((0, 4.5, 0))
    z = np.array((0, 0, 1.5))
    p0 = np.array((0., 0., 9-0.75))
    center = np.array((11.5, 11.5))
    
    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)

    # create the start structure
    sg = StartGenerator(slab=template,
                        atom_numbers=atom_numbers,
                        closest_allowed_distances=cd,
                        box_to_place_in=[p0, [a, b, z], center],
                        elliptic=True,
                        cluster=False)

    structure = sg.get_new_candidate()
    return structure

class globalOptim():
    """
    --Input--
    Efun:
    Function that returns energy of a structure given
    atomic positions in the form [x0, y0, x1, y1, ...]
    
    gradfun:
    Function that returns energy of a structure given
    atomic positions in the form [x0, y0, x1, y1, ...]

    MLmodel:
    Model that given training data can predict energy and gradient of a structure.
    Hint: must include a fit, predict_energy and predict_force methods.
    
    Natoms:
    Number of atoms in structure.
    
    Niter:
    Number of monte-carlo steps.

    boxsize:
    Side length of the square in which the atoms are confined.

    dmax:
    Max translation of each coordinate when perturbing the current structure to
    form a new candidate.
    
    sigma:
    Variable controling how likly it is to accept a worse structure.
    Hint: Should be on the order of the energy difference between local minima.

    Nstag:
    Max number of iterations without accepting new structure before
    the search is terminated.

    saveStep:
    if saveStep=3, every third point in the relaxation trajectory is used for training, and so on..

    min_saveDifference:
    Defines the energy which a new trajectory point has to be lover than the previous, to be saved for training.
    
    MLerrorMargin:
    Maximum error differende between the ML-relaxed structure and the target energy of the same structure,
    below which the ML structure is accepted.

    NstartML:
    The Number of training data required for the ML-model to be used.

    maxNtrain:
    The maximum number of training data. When above this, some of the oldest training data is removed.

    fracPerturb:
    The fraction of the atoms which are ratteled to create a new structure.

    radiusRange:
    Range [rmin, rmax] constraining the initial and perturbed structures. All atoms need to be atleast a
    distance rmin from each other, and have atleast one neighbour less than rmax away.

    """
    def __init__(self, calculator, traj_namebase, MLmodel=None, Natoms=6, Niter=50, std_rattle=0.1, kbT=1, Nstag=10,
                 min_saveDifference=0.3, minSampleStep=10, MLerrorMargin=0.1, NstartML=20, maxNtrain=1.5e3,
                 fracPerturb=0.4):

        self.calculator = calculator
        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel

        self.Natoms = Natoms
        self.std_rattle = std_rattle
        self.Niter = Niter
        self.kbT = kbT
        self.Nstag = Nstag
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.MLerrorMargin = MLerrorMargin
        self.NstartML = NstartML
        self.maxNtrain = int(maxNtrain)
        self.Nperturb = max(2, int(np.ceil(self.Natoms*fracPerturb)))

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0
        
    def runOptimizer(self):
        # Initial structure
        a_init = createInitalStructure2d(self.Natoms)
        self.a, self.E = self.relax(a_init, ML=False)

        # Initialize the best structure
        self.a_best = self.a.copy()
        self.Ebest = self.E.copy()
        
        # Run global search
        stagnation_counter = 0
        for i in range(self.Niter):
            # Perturb current structure
            a_new_unrelaxed = self.makeNewCandidate(self.a)
            
            # Use MLmodel - if it excists + sufficient data is available
            useML_cond = self.MLmodel is not None and self.ksaved > self.NstartML
            if useML_cond:
                # Train ML model if new data is available
                if len(self.a_add) > 0:
                    self.trainModel()

                # Relax with MLmodel
                a_new, EnewML = self.relax(a_new_unrelaxed, ML=True)

                # Singlepoint with objective potential
                Enew = self.singlePoint(a_new)

                """
                # Accept ML-relaxed structure based on precision criteria
                if abs(EnewML - Enew) < self.MLerrorMargin:
                else:
                    continue
                """
            else:
                # Relax with true potential
                a_new, Enew = self.relax(a_new_unrelaxed, ML=False)

            dE = Enew - self.E
            if dE <= 0:  # Accept better structure
                self.E = Enew
                self.a = a_new.copy()
                stagnation_counter = 0
                if Enew < self.Ebest:  # Update the best structure
                    self.Ebest = Enew
                    self.a_best = a_new.copy()
            else:
                p = np.random.rand()
                if p < np.exp(-dE/self.kbT):  # Accept worse structure
                    self.E = Enew
                    self.a = a_new.copy()
                    stagnation_counter = 0
                else:  # Reject structure
                    stagnation_counter += 1

            if stagnation_counter >= self.Nstag:  # The search has converged or stagnated.
                print('The convergence/stagnation criteria was reached')
                break
            
    def makeNewCandidate(self, a):
        a_new = a.copy()
        a_new.set_constraint([FixedPlane(ai.index, (0,0,1)) for ai in a_new])
        a_new.rattle(self.std_rattle)
        return a_new
    
    def trainModel(self):
        """
        # Reduce training data - If there is too much
        if self.ksaved > self.maxNtrain:
            Nremove = self.ksaved - self.maxNtrain
            self.ksaved = self.maxNtrain
            self.MLmodel.remove_data(Nremove)
        """
        GSkwargs = {'reg': np.logspace(-7, -7, 1), 'sigma': np.logspace(0, 2, 5)}
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         **GSkwargs)
        self.a_add = []

    def add_trajectory_to_training(self, trajectory_file):
        atoms = read(filename=trajectory_file, index=':')
        E = [a.get_potential_energy() for a in atoms]
        Nstep = len(atoms)

        # Always add start structure
        self.a_add.append(atoms[0])

        n_last = 0
        Ecurrent = E[0]
        for i in range(1,Nstep):
            n_last += 1
            if Ecurrent - E[i] > self.min_saveDifference and n_last > 10:
                self.a_add.append(atoms[i])
                Ecurrent = E[i]
                self.ksaved += 1
                n_last = 0
        
    def relax(self, a, ML=False):
        if ML:
            traj_name = self.traj_namebase + 'ML{}.traj'.format(self.traj_counter)
            krr_calc = krr_calculator(self.MLmodel)
            a.set_calculator(krr_calc)
            dyn = BFGS(a, trajectory=traj_name)
            dyn.run(fmax=0.1)
        else:
            traj_name = self.traj_namebase + '{}.traj'.format(self.traj_counter)
            a.set_calculator(self.calculator)
            dyn = BFGS(a, trajectory=traj_name)
            dyn.run(fmax=0.1)

            self.add_trajectory_to_training(traj_name)

        self.traj_counter += 1
        Erelaxed = a.get_potential_energy()
        return a, Erelaxed
            
    def singlePoint(self, a):
        a.set_calculator(self.calculator)
        E = a.get_potential_energy()
        results = {'energy': E}
        calc = SinglePointCalculator(a, **results)
        a.set_calculator(calc)
        self.a_add.append(a)
        self.ksaved += 1
        return E

if __name__ == '__main__':
    from custom_calculators import doubleLJ_calculator
    from gaussComparator import gaussComparator
    from angular_fingerprintFeature import Angular_Fingerprint
    from krr_ase import krr_class
    
    Natoms = 10
    
    # Set up featureCalculator
    a = createInitalStructure2d(Natoms)

    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 10
    use_angular = False
    
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
    
    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    traj_namebase = 'globalTest/global'

    
    optimizer = globalOptim(calculator=doubleLJ_calculator(),
                            traj_namebase=traj_namebase,
                            MLmodel=krr,
                            Natoms=Natoms,
                            std_rattle=0.2)

    optimizer.runOptimizer()
