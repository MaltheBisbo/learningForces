import numpy as np
from ase import constraints

class ImitateP4symm():
    """
    Atom index stored as 'a', as is standard in the constraint classes
    """
    def __init__(self, a, a_base, center_xy, theta):
        self.a = a
        self.a_base = a_base
        self.center_xy = center_xy
        self.theta = theta
        c, s = np.cos(theta), np.sin(theta)
        self.R = np.array(((c,-s), (s, c)))
        
    def adjust_positions(self, atoms, newpositions):
        pos_base = newpositions[self.a_base]
        newpositions[self.a] = self.get_p4pos(pos_base)

    def adjust_forces(self, atoms, forces):
        force = forces[self.a_base]
        force[:2] = force[:2] @ self.R
        forces[self.a] = force

    def __repr__(self):
        return 'FixP4symm(%d, (%d) %d/2pi)' % (self.a, self.a_base, int(self.theta/(0.5*np.pi)))

    def get_p4pos(self, pos):
        pos_rot = pos.copy()
        pos_rot[:2] = (pos[:2] - self.center_xy) @ self.R + self.center_xy
        return pos_rot

    def todict(self):
        return {'name': 'ImitateP4symm',
                'kwargs': {'a': self.a, 'a_base': self.a_base, 'center_xy': self.center_xy, 'theta': self.theta}}
"""
    def get_p4pos(self, pos):
        pos_rot = pos.copy()
        pos_xy = pos[:2] - self.center_xy

        pos_xy_rot = pos_xy @ self.R
        pos_xy_rot += self.center_xy

        pos_rot[:2] = pos_xy_rot
        return pos_rot
"""

class ImitateZ():
    """
    Atom index stored as 'a', as is standard in the constraint classes
    """
    def __init__(self, a, a_base):
        self.a = a
        self.a_base = a_base
        
    def adjust_positions(self, atoms, newpositions):
        newpositions[self.a, 2] = newpositions[self.a_base, 2]

    def adjust_forces(self, atoms, forces):
        forces[self.a, 2] = forces[self.a_base, 2]

    def __repr__(self):
        return 'FixP4symm(%d, (%d))' % (self.a, self.a_base)

    def todict(self):
        return {'name': 'ImitateZ',
                'kwargs': {'a': self.a, 'a_base': self.a_base}}
