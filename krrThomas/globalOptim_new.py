import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import time
# import matplotlib.pyplot as plt


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
    def __init__(self, Efun, gradfun, MLmodel=None, Natoms=6, Niter=50, boxsize=None, dmax=0.1, sigma=1, Nstag=10,
                 saveStep=3, min_saveDifference=0.1, MLerrorMargin=0.1, NstartML=20, maxNtrain=1e3,
                 fracPerturb=0.4, radiusRange=[0.9, 1.5], stat=False):

        self.Efun = Efun
        self.gradfun = gradfun
        self.MLmodel = MLmodel
        self.Natoms = Natoms
        if boxsize is not None:
            self.boxsize = boxsize
        else:
            self.boxsize = 1.5 * np.sqrt(self.Natoms)
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
        self.rmin, self.rmax = radiusRange

        # Initialize arrays to store structures for model training
        self.Xsaved = np.zeros((6000, 2*Natoms))
        self.Esaved = np.zeros(6000)
        # initialize counter to keep track of the ammount of data saved
        self.ksaved = 0

        # Initialize array containing Energies of all relaxed structures
        self.Erelaxed = np.zeros(Niter)

        # Initialize counters for function evaluations
        self.Nfev = 0
        self.Nfev_array = np.zeros(Niter)

        # Counters for timing
        self.time_relaxML = 0
        self.time_train = 0

        # Save number of accepted ML-relaxations
        self.NacceptedML = 0

        # ML and target energy of relaxed structures
        self.ErelML = np.zeros(Niter)
        self.ErelTrue = np.zeros(Niter)
        
        # For extracting statistics (Only for testing)
        self.stat = stat
        if stat:
            self.initializeStatistics()

    def runOptimizer(self):
        self.makeInitialStructure()
        self.Ebest = self.E
        self.Xbest = self.X
        k = 0
        
        # Save stuff for performance curve
        self.Erelaxed[0] = self.E
        self.Nfev_array[0] = self.Nfev

        # Run global search
        for i in range(self.Niter):
            
            # Use MLmodel to relax - if it excists
            # and there is sufficient training data
            if self.MLmodel is not None and self.ksaved > self.NstartML:
                
                # Reduce training data - If there is too much
                if self.ksaved > 1.1*self.maxNtrain:
                    
                    Nremove = self.ksaved - self.maxNtrain
                    self.Xsaved[:self.maxNtrain] = self.Xsaved[Nremove:self.ksaved]
                    self.Esaved[:self.maxNtrain] = self.Esaved[Nremove:self.ksaved]
                    self.ksaved = self.maxNtrain
                    
                    self.MLmodel.remove_data(Nremove)
                        
                # Train ML model + ML-relaxation
                t0 = time.time()
                self.trainModel()
                self.time_train += time.time() - t0

                # Try perturbations until acceptable ML-relaxed structure is found
                acceptabelStructure = False
                while not acceptabelStructure:
                    # Perturbation
                    Enew_unrelaxed, Xnew_unrelaxed = self.makeNewCandidate()

                    # ML-relaxation
                    t0 = time.time()
                    EnewML, XnewML = self.relax(Xnew_unrelaxed, ML=True)
                    self.time_relaxML += time.time() - t0

                    acceptabelStructure = self.structureValidity(XnewML)

                    if acceptabelStructure:
                        # Target energy of relaxed structure
                        EnewML_true = self.Efun(XnewML)
                        self.Nfev += 1
                        
                        # Save ML and target energy of relaxed structure (For testing)
                        self.ErelML[i] = EnewML
                        self.ErelTrue[i] = EnewML_true
                        
                        # Save target energy
                        self.Xsaved[self.ksaved] = XnewML
                        self.Esaved[self.ksaved] = EnewML_true
                        self.ksaved += 1
                    
                # Accept ML-relaxed structure based on precision criteria
                if abs(EnewML - EnewML_true) < self.MLerrorMargin:
                    Enew, Xnew = EnewML_true, XnewML
                    self.NacceptedML += 1
                else:
                    continue
                    #Enew, Xnew = self.relax(Xnew_unrelaxed)
            else:
                # Perturb current structure to make new candidate
                Enew_unrelaxed, Xnew_unrelaxed = self.makeNewCandidate()

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
            
    def makeInitialStructure(self):
        """
        Generates an initial structure by placing an atom one at a time inside
        a square of size self.boxsize and cheking if it satisfies the constraints
        based on the previously placed atoms.
        
        Constraints:
        1) Distance to all atoms must be > self.rmin
        2) Distance to nearest neighbor must be < self.rmax
        """
        # Check if position of a new atom is valid with respect to
        # the preciously placed atoms.
        def validPosition(X, xnew):
            Natoms = int(len(X)/2) # Current number of atoms
            if Natoms == 0:
                return True
            connected = False
            for i in range(Natoms):
                r = np.linalg.norm(xnew - X[2*i:2*i+2])
                if r < self.rmin:
                    return False
                if r < self.rmax:
                    connected = True
            return connected

        # Iteratively place a new atom into the box
        Xinit = np.zeros(2*self.Natoms)
        for i in range(self.Natoms):
            while True:
                xnew = np.random.rand(2) * self.boxsize
                if validPosition(Xinit[:2*i], xnew):
                    Xinit[2*i:2*i+2] = xnew
                    break

        self.E, self.X = self.relax(Xinit)
        
    def makeNewCandidate(self):
        """
        Makes a new candidate by perturbing a subset of the atoms in the
        current structure.

        The perturbed atoms are added to the fixed atoms one at a time.

        Pertubations are applied to an atom until it satisfies the
        constraints with respect to the fixed atoms and the previously
        placed perturbed atoms.

        Constraints:
        1) Distance to all atoms must be > self.rmin
        2) Distance to nearest neighbor must be < self.rmax
        """
                
        # Function to determine if perturbation of a single atom is valid
        # with respect to a set of static atoms.
        def validPerturbation(X, index, perturbation, index_static):
            connected = False
            xnew = X[2*index:2*index+2] + perturbation
            for i in index_static:
                if i == index:
                    continue
                r = np.linalg.norm(xnew - X[2*i:2*i+2])
                if r < self.rmin:
                    return False
                if r < self.rmax:
                    connected = True
            return connected

        """
        # Check if unperturbed structure satisfies constraints
        InitialStructureOkay = np.array([validPerturbation(self.X, i, np.array([0,0]), np.arange(self.Natoms))
                                         for i in range(self.Natoms)])
        if not np.all(InitialStructureOkay):
            print('Einit:', self.Efun(self.X))
            assert np.all(InitialStructureOkay)
        """

        # Pick atoms to perturb
        i_permuted = np.random.permutation(self.Natoms)
        i_perturb = i_permuted[:self.Nperturb]
        i_static = i_permuted[self.Nperturb:]
        Xperturb = self.X.copy()
        for i in i_perturb:
            # Make valid perturbation on this atom
            while True:
                perturbation = 2*self.dmax * (np.random.rand(2) - 0.5)
            
                # check if perturbation rersult in acceptable structure
                if validPerturbation(Xperturb, i, perturbation, i_static):
                    Xperturb[2*i:2*i+2] += perturbation
                    # Add the perturbed atom just accepted to the static set
                    i_static = np.append(i_static, i)
                    break

        # Calculate target energy
        Eperturb = self.Efun(Xperturb)
        self.Nfev += 1

        # Save structure for training
        self.Xsaved[self.ksaved] = Xperturb
        self.Esaved[self.ksaved] = Eperturb
        self.ksaved += 1

        return Eperturb, Xperturb

    def structureValidity(self, x):
        connected_array = np.zeros(self.Natoms).astype(int)
        for i in range(self.Natoms):
            xi = x[2*i:2*(i+1)]
            for j in range(i+1, self.Natoms):
                xj = x[2*j:2*(j+1)]
                distance = euclidean(xi, xj)
                if distance < self.rmin:
                    return False
                if distance < self.rmax:
                    connected_array[i] = 1
                    connected_array[j] = 1
        return np.all(connected_array == 1)
    
    def trainModel(self):
        
        Eadd = self.Esaved[self.MLmodel.Ndata:self.ksaved]
        Xadd = self.Xsaved[self.MLmodel.Ndata:self.ksaved]

        GSkwargs = {'reg': np.logspace(-7, -7, 1), 'sigma': np.logspace(0, 2, 5)}
        FVU, params = self.MLmodel.train(Eadd,
                                         positionMat=Xadd,
                                         add_new_data=True,
                                         **GSkwargs)
        """
        FVU, params = self.MLmodel.train(self.Esaved[:self.ksaved],
                                         positionMat=self.Xsaved[:self.ksaved],
                                         add_new_data=False,
                                         **GSkwargs)
        """
        #print('sigma:', params['sigma'], 'reg:', params['reg'])
        
    def relax(self, X=None, ML=False):
        ## determine which model to use for relaxation ##
        if ML:
            # Use ML potential and forces
            def Efun(pos):
                return self.MLmodel.predict_energy(pos=pos)  # + self.artificialPotential(pos)

            def gradfun(pos):
                return -self.MLmodel.predict_force(pos=pos)

            # Set up minimizer
            options = {'gtol': 1e-5}
            def localMinimizer(X):
                res = minimize(Efun, X, method="BFGS", jac=gradfun, tol=1e-5)
                return res
        else:
            # Use double Lennard-Johnes
            # Set up Minimizer
            def callback(x_cur):
                global Xtraj
                Xtraj.append(x_cur)

            def localMinimizer(X):
                global Xtraj
                Xtraj = []
                res = minimize(self.Efun, X, method="BFGS", jac=self.gradfun, tol=1e-5, callback=callback)
                Xtraj = np.array(Xtraj)
                return res, Xtraj

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
            trimIndices  = trimData(Etraj)
            Xtrim = Xtraj[trimIndices]
            Etrim = Etraj[trimIndices]

            # Save new training data
            Ntrim = len(trimIndices)
            # print('right:', Xtrim.shape, 'left:', self.Xsaved[self.ksaved : self.ksaved + Ntrim].shape)
            # print('Ntrim:', Ntrim, 'ksaved:', self.ksaved)
            self.Xsaved[self.ksaved : self.ksaved + Ntrim] = Xtrim
            self.Esaved[self.ksaved : self.ksaved + Ntrim] = Etrim
            self.ksaved += Ntrim

            # Count number of function evaluations
            self.Nfev += res.nfev

            return res.fun, res.x
        else:
            # Minimize using ML potential
            res = localMinimizer(X)
            return res.fun, res.x

    def artificialPotential(self, x):
        N = x.shape[0]
        Natoms = int(N/2)
        x = np.reshape(x, (Natoms, 2))
        E = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                r = np.sqrt(np.dot(x[i] - x[j], x[i] - x[j]))
                if r < self.rmin:
                    E += 1e4 * (self.rmin - r)
        return E

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
