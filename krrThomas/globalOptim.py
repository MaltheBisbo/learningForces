import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ
import matplotlib.pyplot as plt

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
    """
    def __init__(self, Efun, gradfun, MLmodel=None, Natoms=6, Niter=50, boxsize=None, dmax=0.1, sigma=1, Nstag=5, maxIterLocal=10,
                 fracPerturb=0.2):
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
        self.maxIterLocal = maxIterLocal
        self.Nperturb = max(2, int(self.Natoms*fracPerturb))

        # Initialize arrays to store structures for model training
        self.Xsaved = np.zeros((2000, 2*Natoms))
        self.Esaved = np.zeros(2000)
        # initialize index to keep track of the ammount of data saved
        self.ksaved = 0
        
        ## Statistics ##
        # function evaluations
        self.Nfeval = 0
        
    def runOptimizer(self):
        self.makeInitialStructure()
        self.Ebest = self.E
        self.Xbest = self.X
        k = 0
        for i in range(self.Niter):
            Enew, Xnew = self.makeNewCandidate()
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
                    print(p, np.exp(-dE/self.sigma))
                else:  # Decline structure
                    k += 1
            
            if k >= self.Nstag:  # The method search has converged or stagnated.
                print('The convergence/stagnation criteria was reached')
                break
            print('E=', self.E)
        
    def makeInitialStructure(self):
        self.X = np.random.rand(2 * self.Natoms) * self.boxsize
        self.E, self.X = self.relax(self.X)
        
    def makeNewCandidate(self):
        """
        Makes a new candidate by perturbing current structure and
        relaxing the resulting structure.
        """
        # Pick atoms to perturb
        i_perturb = np.random.permutation(self.Natoms)[:self.Nperturb]
        i_perturb.sort()
        Xperturb = self.X.copy()
        for i in i_perturb:
            Xperturb[i:i+2] += 2*self.dmax * (np.random.rand(2) - 0.5)
        Enew, Xnew = self.relax(X=Xperturb)
        return Enew, Xnew

    #def trainModel(self):
        
        
    def relax(self, X=None, ML=False):
        # determine which model to use for potential:
        if ML:
            # Use ML potential and forces
            Efun = doubleLJ_energy
            gradfun = soubleLJ_gradient 
        else:
            # Use double Lennard-Johnes
            Efun = self.Efun
            gradfun = self.gradfun

        # Set up local minimizer
        options = {'maxiter': self.maxIterLocal}  # , 'disp': True}
        
        def localMinimizer(X, func=Efun, bounds=self.bounds, options=options):
            res = minimize(func, X, method="L-BFGS-B", jac=gradfun, tol=1e-3,
                           bounds=bounds, options=options)
            return res

        # Run Local minimization
        if ML is False:
            X0 = X.copy()
            for i in range(100):
                res = localMinimizer(X0)
                print('iterations:', res.nit, '#f eval:', res.nfev)
                self.Xsaved[self.ksaved] = res.x
                self.Esaved[self.ksaved] = res.fun
                self.ksaved += 1
                if res.success:
                    break
                X0 = res.x
            return self.Esaved[self.ksaved-1], self.Xsaved[self.ksaved-1]
        else:
            # Need to use the ML potential and force
            print('hello')
            res = localMinimizer(X)
            return res.fun, res.x

    def plotCurrentStructure(self):
        xbox = np.array([0, self.boxsize, self.boxsize, 0, 0])
        ybox = np.array([0, 0, self.boxsize, self.boxsize, 0])

        x = self.X[0::2]
        y = self.X[1::2]
        plt.plot(xbox, ybox, color='k')
        plt.scatter(x, y, s=8)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
