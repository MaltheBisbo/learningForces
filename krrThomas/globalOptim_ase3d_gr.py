import numpy as np

from startgenerator import StartGenerator
from custom_calculators import krr_calculator

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.ga.utilities import closest_distances_generator
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak

import ase.parallel as mpi
world = mpi.world


def relaxGPAW(structure, label):
    '''
    Relax a structure and saves the trajectory based in the index i

    Parameters
    ----------
    structure : ase Atoms object to be relaxed

    i : index which the trajectory is saved under

    ranks: ranks of the processors to relax the structure 

    Returns
    -------
    structure : relaxed Atoms object
    '''

    # Create calculator
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),
              occupations=FermiDirac(0.1),
              maxiter=99,
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              kpts=(1,1,1),
              txt = label+ '_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)

    # loop a number of times to capture if minimization stops with high force
    # due to the VariansBreak calls
    forcemax = 0.1
    niter = 0

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    traj = Trajectory(label+'_lcao.traj','a')
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < 10:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(traj)
        dyn.attach(vb)
        dyn.run(fmax = forcemax, steps = 500)
        niter += 1

    return structure

def singleGPAW(structure, label):
    # Create calculator
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),
              occupations=FermiDirac(0.1),
              maxiter=99,
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              kpts=(1,1,1),
              txt = label+'_single_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)
    
    return structure.get_potential_energy()

def rattle_atom2d(struct, index_rattle, rmax_rattle=1.0, rmin=0.9, rmax=1.7, Ntries=10):
    Natoms = struct.get_number_of_atoms()
    
    structRattle = struct.copy()
    mindis = 0
    mindisAtom = 10

    for i in range(Ntries):
        # First load original positions
        positions = struct.positions.copy()
        
        # Then Rattle within a circle
        r = rmax_rattle * np.sqrt(np.random.rand())
        theta = np.random.uniform(low=0, high=2*np.pi)
        positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])

        structRattle.positions = positions
        dis = structRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])  # Check that we are not too close
        mindisAtom = np.min(structRattle.get_distances(index_rattle, np.delete(np.arange(Natoms), index_rattle)))  # check that we are not too far
        
        # If it does not fit, try to wiggle it into place using small circle
        if mindis < rmin:
            for i in range(10):
                r = 0.5 * np.sqrt(np.random.rand())
                theta = np.random.uniform(low=0, high=2 * np.pi)
                positions[index_rattle] += r * np.array([np.cos(theta), np.sin(theta), 0])
                structRattle.positions = positions
                dis = structRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > rmin:
                    break

                # Otherwise reset coordinate
                else:
                    positions[index_rattle] -= r * np.array([np.cos(theta), np.sin(theta), 0])

        # STOP CRITERION
        if mindis > rmin and mindisAtom < rmax:
            return structRattle
    
    # Return None if no acceptable rattle was found
    return None


def rattle_Natoms2d(struct, Nrattle, rmax_rattle=1.0, rmin=0.9, rmax=1.7, Ntries=10):
    structRattle = struct.copy()
    
    Natoms = struct.get_number_of_atoms()
    i_permuted = np.random.permutation(Natoms)

    rattle_counter = 0
    for index in i_permuted:
        newStruct = rattle_atom2d(structRattle, index, rmax_rattle, rmin, rmax, Ntries)
        if newStruct is not None:
            structRattle = newStruct.copy()
            rattle_counter += 1

        # The desired number of rattles have been performed
        if rattle_counter > Nrattle:
            return structRattle

    # The desired number of succesfull rattles was not reached
    return structRattle


def createInitalStructure(Natoms):
    '''
    Creates an initial structure of 24 Carbon atoms
    '''    
    number_type1 = 6  # Carbon
    number_opt1 = Natoms  # number of atoms
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
    calculator:
    ASE calculator for calculating the true energy and forces

    MLmodel:
    Model that given training data can predict energy and gradient of a structure.
    Hint: must include a fit, predict_energy and predict_force methods.
    
    Natoms:
    Number of atoms in structure.
    
    Niter:
    Number of monte-carlo steps.

    rattle_maxDist:
    Max translation of each coordinate when perturbing the current structure to
    form a new candidate.
    
    kbT:
    Variable controling how likly it is to accept a worse structure.
    Hint: Should be on the order of the energy difference between local minima.

    Nstag:
    Max number of iterations without accepting new structure before
    the search is terminated.

    min_saveDifference:
    Defines the energy which a new trajectory point has to be lover than the previous, to be saved for training.
    
    minSampleStep:
    if minSampleStep=10, every tenth point in the relaxation trajectory is used for training, and so on..
    Unless min_saveDifference have not been ecxeeded.

    MLerrorMargin:
    Maximum error differende between the ML-relaxed structure and the target energy of the same structure,
    below which the ML structure is accepted.

    NstartML:
    The Number of training data required for the ML-model to be used.

    maxNtrain:
    The maximum number of training data. When above this, some of the oldest training data is removed.

    radiusRange:
    Range [rmin, rmax] constraining the initial and perturbed structures. All atoms need to be atleast a
    distance rmin from each other, and have atleast one neighbour less than rmax away.

    fracPerturb:
    The fraction of the atoms which are ratteled to create a new structure.

    noZ:
    Atoms are not mooved in the z-direction during relaxation.
    """
    def __init__(self, calculator, traj_namebase, MLmodel=None, Natoms=6, Niter=50, rattle_maxDist=0.5, kbT=1, Nstag=10,
                 min_saveDifference=0.3, minSampleStep=10, MLerrorMargin=0.1, NstartML=20, maxNtrain=1.5e3,
                 radiusRange=[0.9, 1.5], fracPerturb=0.3, noZ=False):

        self.calculator = calculator
        self.traj_namebase = traj_namebase
        self.MLmodel = MLmodel

        self.Natoms = Natoms
        self.rattle_maxDist = rattle_maxDist
        self.Niter = Niter
        self.kbT = kbT
        self.Nstag = Nstag
        self.min_saveDifference = min_saveDifference
        self.minSampleStep = minSampleStep
        self.MLerrorMargin = MLerrorMargin
        self.NstartML = NstartML
        self.maxNtrain = int(maxNtrain)
        self.rmin, self.rmax = radiusRange
        self.Nperturb = max(2, int(np.ceil(self.Natoms*fracPerturb)))
        self.noZ = noZ

        # List of structures to be added in next training
        self.a_add = []

        self.traj_counter = 0
        self.ksaved = 0

        self.master = world.rank == 0
        

    def runOptimizer(self):
        # Initial structure
        a_init = createInitalStructure(self.Natoms)
        self.a, self.E = self.relax(a_init, ML=False)

        # Initialize the best structure
        if self.master:
            self.a_best = self.a.copy()
            self.Ebest = self.E
        
        # Initialize a_new
        a_new = self.a.copy()

        # Run global search
        stagnation_counter = 0
        for i in range(self.Niter):
            # Perturb current structure
            positions_unrelaxed = np.zeros((Natoms, 3))
            if self.master:
                a_new = rattle_Natoms2d(struct=self.a,
                                        Nrattle=self.Nperturb,
                                        rmax_rattle=self.rattle_maxDist,
                                        rmin=self.rmin,
                                        rmax=self.rmax)
                positions_unrelaxed = a_new.positions
            world.broadcast(positions_unrelaxed, 0)
            a_new.positions = positions_unrelaxed
            world.barrier()
            
            # Use MLmodel - if it excists + sufficient data is available
            useML_cond = self.MLmodel is not None and self.ksaved > self.NstartML
            if useML_cond:
                positions_MLrelaxed = np.zeros((Natoms, 3))
                if self.master:
                    print("Begin training")
                    # Train ML model if new data is available
                    if len(self.a_add) > 0:
                        self.trainModel()
                    
                    # Relax with MLmodel
                    a_new, EnewML = self.relax(a_new, ML=True)
                    positions_MLrelaxed = a_new.positions
                world.broadcast(positions_MLrelaxed, 0)
                a_new.positions = positions_MLrelaxed
                world.barrier()

                # Singlepoint with objective potential
                Enew = self.singlePoint(a_new)
                if self.master:
                    print('True energy of relaxed structure:', Enew)
            else:
                # Relax with true potential
                a_new, Enew = self.relax(a_new, ML=False)
            
            if self.master:
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
        for i in range(1, Nstep):
            n_last += 1
            if Ecurrent - E[i] > self.min_saveDifference and n_last > 10:
                self.a_add.append(atoms[i])
                Ecurrent = E[i]
                self.ksaved += 1
                n_last = 0
        
    def relax(self, a, ML=False):
        # Trajectory name
        if len(str(self.traj_counter)) == 1:
            traj_counter = '00' + str(self.traj_counter)
        if len(str(self.traj_counter)) == 2:
            traj_counter = '0' + str(self.traj_counter)
        
        # Relax with ML or target potential
        if ML:
            traj_name = self.traj_namebase + '{}_ML.traj'.format(traj_counter)
            krr_calc = krr_calculator(self.MLmodel, noZ=self.noZ)
            a.set_calculator(krr_calc)
            dyn = BFGS(a, trajectory=traj_name)
            dyn.run(fmax=0.1)
        else:
            # Broadcast structure to all cores
            positions = a.positions
            if self.master:
                positions = a.positions
            world.broadcast(positions, 0)
            a.positions = positions
            world.barrier()
            
            label = self.traj_namebase + '{}'.format(traj_counter)
            a = relaxGPAW(a, label)

            traj_name = label+'_lcao.traj'
            self.add_trajectory_to_training(traj_name)

        self.traj_counter += 1
        Erelaxed = a.get_potential_energy()
        return a, Erelaxed
            
    def singlePoint(self, a):
        # Trajectory name
        if len(str(self.traj_counter)) == 1:
            traj_counter = '00' + str(self.traj_counter)
        if len(str(self.traj_counter)) == 2:
            traj_counter = '0' + str(self.traj_counter)

        label = self.traj_namebase + '{}'.format(traj_counter)

        positions = a.positions
        if self.master:
            positions = a.positions
        world.broadcast(positions, 0)
        a.positions = positions
        world.barrier()
        
        E = singleGPAW(a, label)
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
    
    Natoms = 13
    
    # Set up featureCalculator
    a = createInitalStructure(Natoms)

    Rc1 = 5
    binwidth1 = 0.2
    sigma1 = 0.2
    
    Rc2 = 4
    Nbins2 = 30
    sigma2 = 0.2
    
    gamma = 1
    eta = 20
    use_angular = True
    
    featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
    
    # Set up KRR-model
    comparator = gaussComparator()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    traj_namebase = 'grendel/DFTrelax2/global'

    optimizer = globalOptim(calculator=doubleLJ_calculator(noZ=True),
                            traj_namebase=traj_namebase,
                            MLmodel=krr,
                            Natoms=Natoms,
                            Niter=50,
                            NstartML=10,
                            fracPerturb=0.4,
                            rattle_maxDist=1,
                            noZ=True)

    optimizer.runOptimizer()
