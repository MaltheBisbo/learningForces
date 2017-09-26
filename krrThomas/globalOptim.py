import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ
import matplotlib.pyplot as plt

class globalOptim():
    """
    --Input--
    Efun:
    Function that returns energy and gradient of a structure given
    atomic positions in the form [x0, y0, x1, y1, ...]

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
    def __init__(self, Efun, Natoms=6, Niter=50, boxsize=None, dmax=0.1, sigma=1, Nstag=5):
        self.Efun = Efun
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
        Xperturb = self.X + 2*self.dmax * (np.random.rand(2*self.Natoms) - 0.5)
        Enew, Xnew = self.relax(X=Xperturb)
        return Enew, Xnew
        
    def relax(self, X=None, ML=False, maxiter=None):
        if X is None:
            X = self.X
        if ML:
            # Use ML potential and forces
            func = doubleLJ
        else:
            # Use double Lennard-Johnes
            func = self.Efun
            
        options = {'maxiter': maxiter}  # , 'disp': True}

        def localMinimizer(X, func=func, bounds=self.bounds, options=options):
            res = minimize(func, X, method="TNC", jac=True, tol=1e-3,
                           bounds=bounds, options=options)
            return res

        if ML is False:
            X0 = X.copy()
            savedStructures = []
            savedEnergies = []
            for i in range(100):
                res = localMinimizer(X0)
                savedStructures.append(res.x)
                savedEnergies.append(res.fun)
                if res.success:
                    break
                X0 = res.x
            return savedEnergies[-1], savedStructures[-1]
        else:
            # Need to use the ML potential and force
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
        
