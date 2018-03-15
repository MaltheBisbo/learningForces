import numpy as np
from scipy.spatial.distance import euclidean

from ase.calculators.calculator import Calculator



class krr_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, MLmodel, label='MLmodel', **kwargs):
        self.MLmodel = MLmodel

        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):

        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = self.MLmodel.predict_energy(atoms)
        self.results['forces'] = self.MLmodel.predict_force(atoms).reshape((-1,3))
        
        return self.results['energy'], self.results['forces']


class doubleLJ_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, eps=1.8, r0=1.1, sigma=np.sqrt(0.02), label='doubleLJ', **kwargs):
        self.eps = eps
        self.r0 = r0
        self.sigma = sigma
        
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = self.energy(atoms)
        self.results['forces'] = self.force(atoms)
        
        return self.results['energy'], self.results['forces']

    def energy(self, a):
        x = a.get_positions()
        E = 0
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                if j > i:
                    r = euclidean(xi, xj)
                    E1 = 1/r**12 - 2/r**6
                    E2 = -self.eps * np.exp(-(r - self.r0)**2 / (2*self.sigma**2))
                    E += E1 + E2
        return E

    def doubleLJ_gradient_ase(self, a):
        x = a.get_positions()
        Natoms, dim = x.shape
        dE = np.zeros((Natoms, dim))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                r = euclidean(xi,xj)
                if j != i:
                    rijVec = xi-xj

                    dE1 = 12*rijVec*(-1 / r**14 + 1 / r**8)
                    dE2 = self.eps*(r-self.r0)*rijVec / (r*self.sigma**2) * np.exp(-(r - self.r0)**2 / (2*self.sigma**2))
                    
                    dE[i] += dE1 + dE2
        return dE
