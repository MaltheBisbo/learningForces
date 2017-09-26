import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ
import matplotlib.pyplot as plt

class globalOptim():
    """
    Documentation
    """
    def __init__(self, Natoms=6, Niter=10, boxsize=None, dmax=0.1, Eparams=(1.8, 1.1, np.sqrt(0.02))):
        self.Natoms = Natoms
        if boxsize is not None:
            self.boxsize = boxsize
        else:
            self.boxsize = 3 * np.sqrt(self.Natoms)
        self.bounds = [(0, boxsize)] * Natoms * 2
        self.dmax = dmax
        self.Niter = Niter
        self.Eparams = Eparams

    def makeInitialStructure(self):
        self.X = np.random.rand(2 * self.Natoms) * self.boxsize
        E, self.X = self.relax(self.X)
    def makeNewCandidate(self):
        """
        Makes a new candidate by perturbing current structure and
        relaxing the resulting structure.
        """

    def relax(self, X=None, ML=False, maxiter=None):
        if X is None:
            X = self.X
        if ML:
            # Use ML potential and forces
            func = doubleLJ
        else:
            # Use double Lennard-Johnes
            func = doubleLJ
            
        options = {'maxites': maxiter}            
        def localMinimizer(X, func=func, params=self.Eparams,
                           bounds=self.bounds, options=options):
            res = minimize(func, X, params,
                           method="TNC",
                           jac=True,
                           tol=1e-3,
                           bounds=bounds)
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
        
