import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import time

from startgenerator import StartGenerator
from custom_calculators import krr_calculator

from ase import Atoms
from ase.ga.utilities import closest_distances_generator

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
    def __init__(self, calculator, MLmodel=None, Natoms=6, Niter=50, dmax=0.1, sigma=1, Nstag=10,
                 saveStep=3, min_saveDifference=0.1, MLerrorMargin=0.1, NstartML=20, maxNtrain=1.5e3,
                 fracPerturb=0.4):

        self.calculator = calculator
        self.MLmodel = MLmodel

        self.Natoms = Natoms
        self.bounds = [(0, boxsize)] * Natoms * 2
        self.dmax = dmax
        self.Niter = Niter
        self.sigma = sigma
        self.Nstag = Nstag
        self.saveStep = saveStep
        self.min_saveDifference = min_saveDifference
        self.MLerrorMargin = MLerrorMargin
        self.NstartML = NstartML
        self.maxNtrain = int(maxNtrain)
        self.Nperturb = max(2, int(np.ceil(self.Natoms*fracPerturb)))

        # List of structures to be added in next training
        self.a_add = []
        
    def runOptimizer(self):
        self.a_best = createInitalStructure()
        self.a_best.set_calculator(self.calculator)
        Ebest = a_best.get_potential_energy()
        k = 0

        # Run global search
        for i in range(self.Niter):
            
            # Use MLmodel - if it excists + sufficient data is available
            useML_cond = self.MLmodel is not None and self.ksaved > self.NstartML
            if useML_cond:
                
                # Reduce training data - If there is too much
                if self.ksaved > self.maxNtrain:
                    Nremove = self.ksaved - self.maxNtrain
                    self.ksaved = self.maxNtrain

                    # Remove data in MLmodel
                    self.MLmodel.remove_data(Nremove)
                        
                # Train ML model + ML-relaxation
                self.trainModel()

                # Perturb current structure
                a_new = self.makeNewCandidate(a_best)
                
                # Relax with MLmodel
                EnewML, XnewML, error, theta0, Nback = self.relax(Xnew_unrelaxed, ML=True)  # two last for TESTING
                
            else:
                # Perturb current structure to make new candidate
                Xnew_unrelaxed = self.makeNewCandidate()

                # Relax with true potential
                Enew, Xnew = self.relax(Xnew_unrelaxed)

            dE = Enew - self.E
            if dE <= 0:  # Accept better structure
                self.E = Enew
                self.X = Xnew
                k = 0
                if Enew < self.Ebest:  # Update the best structure
                    self.Ebest = Enew
                    self.Xbest = Xnew
            else:
                p = np.random.rand()
                if p < np.exp(-dE/self.sigma):  # Accept worse structure
                    self.E = Enew
                    self.X = Xnew
                    k = 0
                else:  # Reject structure
                    k += 1

            # Save stuff for performance curve
            self.Erelaxed[i] = self.E
            self.Nfev_array[i] = self.Nfev
            
            if k >= self.Nstag:  # The search has converged or stagnated.
                print('The convergence/stagnation criteria was reached')
                break

            """
            # Other termination criterias (for testing)
            if self.stat:
                if self.testCounter > self.NtestPoints-1:
                    break
            if self.ksaved > 1500:
                break
            """
            
    def makeNewCandidate(self, a):
        
        a_new = a.copy()
        a_new.rattle()
        a_new.set_calculator(self.calculator)
        
        return a_new
    
    def trainModel(self):
        GSkwargs = {'reg': np.logspace(-7, -7, 1), 'sigma': np.logspace(0, 2, 5)}
        FVU, params = self.MLmodel.train(atoms_list=self.a_add,
                                         add_new_data=True,
                                         **GSkwargs)
        self.a_add = []
        
    def relax(self, a, ML=False):
        ## determine which model to use for relaxation ##
        if ML:
            # Set up krr calculator
            krr_calc = krr_calculator(self.MLmodel)
            a.set_calculator(krr_calc)
            dyn = BFGS(a, trajectory='grapheneMLrelax/grapheneNoZ_ree{}.traj'.format(i))
            dyn.run(fmax=0.1)
        else:
            a.set_calculator(self.calculator)
            dyn = BFGS(a, trajectory='grapheneMLrelax/grapheneNoZ_ree{}.traj'.format(i))
            dyn.run(fmax=0.1)
            

        # Function that extracts the subset, of the relaxation trajectory, relevant for training.
        def trimData(Etraj):
            Nstep = len(Etraj)
            index = []
            k = 0
            Ecur = Etraj[0]
            while k < Nstep:
                if len(index) == 0:
                    index.append(0)
                    continue
                elif Ecur - Etraj[k] > self.min_saveDifference:
                    index.append(k)
                    Ecur = Etraj[k]
                k += self.saveStep
            index[-1] = Nstep - 1
            return index
            
        ## Run Local minimization ##
        if ML is False:
            # Relax
            res, Xtraj = localMinimizer(X)
            Etraj = np.array([self.Efun(x) for x in Xtraj])

            # Extract subset of trajectory for training
            trimIndices = trimData(Etraj)
            Xtrim = Xtraj[trimIndices]
            Etrim = Etraj[trimIndices]

            # Save new training data
            Ntrim = len(trimIndices)
            # print('right:', Xtrim.shape, 'left:', self.Xsaved[self.ksaved : self.ksaved + Ntrim].shape)
            # print('Ntrim:', Ntrim, 'ksaved:', self.ksaved)
            self.Xsaved[self.ksaved:self.ksaved + Ntrim] = Xtrim
            self.Esaved[self.ksaved:self.ksaved + Ntrim] = Etrim
            self.ksaved += Ntrim

            # Count number of function evaluations
            self.Nfev += res.nfev

            return res.fun, res.x
        else:
            # Minimize using ML potential
            res, Xtraj = localMinimizer(X)
            k = 0
            for x in reversed(Xtraj):
                E, error, theta0 = self.MLmodel.predict_energy(pos=x, return_error=True)
                #if error < 0.95*np.sqrt(theta0):  # 0.5 as first trial (testing)
                return E, x, error, theta0, k  # two last is only for TESTING
                #k += 1
            #self.Nerror_too_high += 1
            #return res.fun, res.x, np.nan, np.nan, np.nan

    def initializeStatistics(self):
        ### Statistics ###
        # Function evaluations
        self.Nfeval = 0
        # Predicted energy of structure relaxed with ML model
        self.ErelML = []
        # Energy of structure relaxed with true potential
        self.ErelTrue = []
        # True energy of resulting from relaxation with ML model
        self.ErelMLTrue = []
        # Energy of structure relaxed with ML model followed by relaxation with true potential
        self.E2rel = []
        # Predicted energy of unrelaxed structure
        self.EunrelML = []
        # Energy of unrelaxed structure
        self.Eunrel = []
        # All force components of unrelaxed structure (ML)
        self.FunrelML = []
        # All force components of unrelaxed structure (True)
        self.FunrelTrue = []
        # The number of training data used
        self.ktrain = []

        self.testCounter = 0
        self.NtestPoints = 10
        self.Ntest_array = np.logspace(1, 3, self.NtestPoints)
        
    def saveStatistics(self):
        print('ksaved=', self.ksaved)
        
        self.testCounter += 1
        self.trainModel()
        self.ktrain.append(self.ksaved)
        # New unrelaxed structure
        Enew_unrelaxed, Xnew_unrelaxed = self.makeNewCandidate()
        # relax with true potential
        Enew, Xnew = self.relax(Xnew_unrelaxed)
        # relax with ML potential
        ErelML, XrelML = self.relax(Xnew_unrelaxed, ML=True)
        ErelML_relax, XrelML_relax = self.relax(XrelML)
        # self.plotStructures(Xnew, XrelML, Xnew_unrelaxed)
        ErelMLTrue = self.Efun(XrelML)
        
        # Data for relaxed energies
        self.ErelML.append(ErelML)
        self.ErelMLTrue.append(ErelMLTrue)
        self.ErelTrue.append(Enew)
        self.E2rel.append(ErelML_relax)
        
        # Data for unrelaxed energies
        self.Eunrel.append(Enew_unrelaxed)
        self.EunrelML.append(self.MLmodel.predict_energy(pos=Xnew_unrelaxed))
        
        # Data for unrelaxed forces
        FnewML = self.MLmodel.predict_force(pos=Xnew_unrelaxed)
        FnewTrue = -self.gradfun(Xnew_unrelaxed)
        self.FunrelML.append(FnewML)
        self.FunrelTrue.append(FnewTrue)
        
        return Enew, Xnew
    """
    def plotStructures(self, X1=None, X2=None, X3=None):
        xbox = np.array([0, self.boxsize, self.boxsize, 0, 0])
        ybox = np.array([0, 0, self.boxsize, self.boxsize, 0])

        plt.gca().cla()
        x1 = X1[0::2]
        y1 = X1[1::2]
        x2 = X2[0::2]
        y2 = X2[1::2]
        x3 = X3[0::2]
        y3 = X3[1::2]
        plt.plot(xbox, ybox, color='k')
        plt.scatter(x1, y1, s=22, color='r')
        plt.scatter(x2, y2, s=22, color='b')
        plt.scatter(x3, y3, s=22, color='g', marker='x')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(2)
    """

    
    """
    options = {'maxiter': self.maxIterLocal}  # , 'gtol': 1e-3}
    def localMinimizer(X):
        res = minimize(self.Efun, X, method="L-BFGS-B", jac=self.gradfun, tol=1e-5,
                       bounds=self.bounds, options=options)
        return res
    """

    """
    if ML is False:
            # Minimize using double-LJ potential + save trajectories
            X0 = X.copy()
            for i in range(100):
                res = localMinimizer(X0)
                # Save training data
                if np.isnan(res.fun):
                    print('NaN value during relaxation')
                if res.fun < 0 and res.fun:  # < self.Esaved[self.ksaved-1] - 0.2:
                    self.Xsaved[self.ksaved] = res.x
                    self.Esaved[self.ksaved] = res.fun
                    self.ksaved += 1
                # print('Number of iterations:', res.nit, 'fev:', res.nfev)
                if res.success: # Converged
                    break
                X0 = res.x
            return self.Esaved[self.ksaved-1], self.Xsaved[self.ksaved-1]
        else:
            # Minimize using ML potential
            res = localMinimizer(X)
            print('Number of ML iterations:', res.nit, 'fev:', res.nfev)
            print('Convergence status:', res.status)
            print(res.message)
            return res.fun, res.x
    """
