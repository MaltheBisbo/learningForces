import numpy as np
from ase.calculators.calculator import Calculator

class krr_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, MLmodel, label, **kwargs):
        self.MLmodel = MLmodel

        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):

        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = self.MLmodel.predict_energy(atoms)
        self.results['forces'] = self.MLmodel.predict_force(atoms).reshape((-1,3))
        
        return self.results['energy'], self.results['forces']
