""" Methods for generating new random starting candidates. """
from random import shuffle
import numpy as np
from ase import Atoms
from ase.ga.utilities import get_mic_distance
from math import *
from copy import deepcopy
from time import sleep
#from builtins import *  # to get range in python 2.7

def random_pos(box):
    """ Returns a random position within the box
         described by the input box. """
    p0 = box[0]
    vspan = box[1]
    r = np.random.random((1, len(vspan)))
    pos = p0.copy()
    for i in range(len(vspan)):
        pos += vspan[i] * r[0, i]
    return pos

def random_pos_elliptic(box):
    """ Returns a random position within the elliptic 
        cylinder described by the input vectors a,b,c 
        and the x,y-coordinates for the center. """
    p0 = box[0]
    vspan = box[1]
    center = box[2]
    a = vspan[0][0].copy()
    b = vspan[1][1].copy()
    tmp_vspan = deepcopy(vspan)
    tmp_vspan[0][0] += tmp_vspan[0][0] * 2
    tmp_vspan[1][1] += tmp_vspan[1][1] * 2
    F1 = center.copy()
    F2 = center.copy()
    F1[0] += sqrt(a ** 2 - b ** 2) 
    F2[0] += -sqrt(a ** 2 - b ** 2) 
    approved = False
    while approved == False:
        r = np.random.random((1, len(vspan)))
        pos = p0.copy()
        for i in range(len(vspan)):
            pos += tmp_vspan[i] * r[0, i]
        pos += np.array((center[0], center[1], 0)) - np.array((a, b, 0))
        PF1 = np.linalg.norm(F1 - (pos[0], pos[1]))
        PF2 = np.linalg.norm(F2 - (pos[0], pos[1]))
        if PF1 + PF2 <= 2 * a:
            approved = True
    return pos



class StartGenerator(object):

    """ Class used to generate random starting candidates.
        The candidates are generated by iteratively adding in
        one atom at a time within the box described.

        Parameters:

        slab: The atoms object describing the super cell to
        optimize within.
        atom_numbers: A list of the atomic numbers that needs
        to be optimized.
        closed_allowed_distances: A dictionary describing how
        close two atoms can be.
        box_to_place_in: The box atoms are placed within. 
        if elliptic is False the standard box for atoms is used:
           The format is [p0, [v1, v2, v3]] with positions being 
           generated as p0 + r1 * v1 + r2 * v2 + r3 + v3
        if elliptic is True the box used to contain the atom distribution:
           The format is [p0, [a, b, z], center] with positions being
           generated and limited to only be inside a elliptic box 
           with p0 and center describing the center and a,b,z the normal
           length, width and height.
        cluster: The attribute that describes whether the atoms 
        can or can't be placed far apart.
    """
    def __init__(self, slab, atom_numbers,
                 closest_allowed_distances, box_to_place_in, elliptic=False, cluster=True):
        self.slab = slab
        self.atom_numbers = atom_numbers
        self.blmin = closest_allowed_distances
        self.box = box_to_place_in
        self.elliptic = elliptic
        self.cluster = cluster

    def get_new_candidate(self,maxlength=2.):
        """ Returns a new candidate. """
        N = len(self.atom_numbers)
        cell = self.slab.get_cell()
        pbc = self.slab.get_pbc()

        # The ordering is shuffled so different atom
        # types are added in random order.
        order = [i for i in range(N)]  # range(N)
        shuffle(order)
        num = [i for i in range(N)]  # range(N)
        for i in range(N):
            num[i] = self.atom_numbers[order[i]]
        blmin = self.blmin

        # Runs until we have found a valid candidate.
        while True:
            pos = np.zeros((N, 3))
            # Make each new position one at a time.
            for i in range(N):
                pos_found = False
                pi = None
                for _ in range(1000):
                    if self.elliptic == True:
                        pi = random_pos_elliptic(self.box)
                    else:
                        pi = random_pos(self.box)
                    if i == 0:
                        pos_found = True
                        break
                    isolated = self.cluster
                    too_close = False
                    for j in range(i):
                        d = get_mic_distance(pi, pos[j], cell, pbc)
                        bij_min = blmin[(num[i], num[j])]
                        bij_max = bij_min * maxlength
                        if d < bij_min:
                            too_close = True
                            break
                        if d < bij_max:
                            isolated = False
                    # A new atom must be near something already there,
                    # but not too close.
                    if not isolated and not too_close:
                        pos_found = True
                        break
                if pos_found:
                    pos[i] = pi
                else:
                    break
            if not pos_found:
                continue

            # Put everything back in the original order.
            pos_ordered = np.zeros((N, 3))
            for i in range(N):
                pos_ordered[order[i]] = pos[i]
            pos = pos_ordered
            top = Atoms(self.atom_numbers, positions=pos, pbc=pbc, cell=cell)

            # At last it is verified that the new cluster is not too close
            # to the slab it is supported on.
            tf = False
            for i in range(len(self.slab)):
                for j in range(len(top)):
                    dmin = blmin[(self.slab.numbers[i], top.numbers[j])]
                    d = get_mic_distance(self.slab.positions[i],
                                         top.positions[j], cell, pbc)
                    if d < dmin:
                        tf = True
                        break
                if tf:
                    break
            if not tf:
                break
        return self.slab + top
