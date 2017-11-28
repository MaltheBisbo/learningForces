import numpy as np
from ase.calculators.calculator import Calculator
from krr_class_new import krr_class

class krr_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, MLmodel, label, **kwargs):
        self.MLmodel = MLmodel

        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):

        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        dim = positions.shape[1]
        positions = positions.reshape(-1)
        
        self.results['energy'] = self.MLmodel.predict_energy(pos=positions)
        self.results['forces'] = self.MLmodel.predict_force(pos=positions).reshape((-1,dim))

        return self.results['energy'], self.results['forces']
