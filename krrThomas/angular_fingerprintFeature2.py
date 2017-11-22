import os
import sys
import numpy as np
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

class Angular_Fingerprint(object):
    """ comparator for construction of angular fingerprints
    """

    def __init__(self, atoms, Rc1=6.5, Rc2=4.0, binwidth1=0.05, binwidth2=0.025, sigma1=0.5, sigma2=0.25, nsigma=4):
        """ Set a common cut of radius
        """
        self.Rc1 = Rc1
        self.Rc2 = Rc2
        self.binwidth1 = binwidth1
        self.binwidth2 = binwidth2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.nsigma = nsigma
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        self.n_atoms = atoms.get_number_of_atoms()
        self.num = atoms.get_atomic_numbers()
        self.atomic_types = sorted(list(set(self.num)))
        self.atomic_count = {type:list(self.num).count(type) for type in self.atomic_types}
        self.volume = atoms.get_volume()
        self.dim = 3

        # parameters for the binning:
        self.m1 = int(np.ceil(self.nsigma*self.sigma1/self.binwidth1))  # number of neighbour bins included.
        self.smearing_norm1 = erf(0.25*np.sqrt(2)*self.binwidth1*(2*self.m1+1)*1./self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = int(np.ceil(self.nsigma*self.sigma2/self.binwidth2))  # number of neighbour bins included.
        self.smearing_norm2 = erf(0.25*np.sqrt(2)*self.binwidth2*(2*self.m2+1)*1./self.sigma2)  # Integral of the included part of the gauss
        self.Nbins2 = int(np.ceil(self.Rc2/self.binwidth2))

        # Cutoff surface areas
        self.cutoff_surface_area1 = 4*np.pi*self.Rc1**2
        self.cutoff_surface_area2 = 4*np.pi*self.Rc2**2
        
        
    def get_features(self, atoms):
        """
        """
        feature = [None]*2
        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_types = self.atomic_types
        atomic_count = self.atomic_count
        volume = self.volume

        nb_distVec, nb_deltaRs, nb_bondtype, np_index = self.__get_neighbour_lists(pos, num, pbc, cell, n_atoms)

        # Initialize fingerprint object
        feature = [{} for _ in range(2)]  # [None]*2

        # two body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                key = tuple(sorted([type1, type2]))
                if key not in feature[0]: 
                    feature[0][key] = np.zeros(self.Nbins1)

        # three body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                for type3 in self.atomic_types:
                    key = tuple([type1] + sorted([type1, type2]))
                    if key not in feature[0]:
                        feature[1][key] = np.zeros(self.Nbins2)
        
        for i in range(n_atoms):
            for n in range(len(nb_distVec[i])):
                (N1,N2) = nb_bondtype[i][n]
                deltaR = nb_deltaRs[i][n]
                rbin = int(np.floor(deltaR/self.binwidth1))
                binpos = (deltaR % self.binwidth1) / self.binwidth1  # From 0 to binwidth (set constant at 0.5*binwidth for original)
                rabove = int(binpos > 0.5)
                
                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = -self.m1-(1-rabove)
                maxbin_lim = self.m1+1+rabove
                for i in range(minbin_lim, maxbin_lim):
                    newbin = rbin + i  # maybe abs() to make negative bins contribute aswell.
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue
                
                    c = 0.25*np.sqrt(2)*self.binwidth1*1./self.sigma1
                    if i == minbin_lim:
                        erfarg_low = -(self.m1+0.5)
                        erfarg_up = i+(1-binpos)
                    elif i == maxbin_lim-1:
                        erfarg_low = i-binpos
                        erfarg_up = self.m1+0.5
                    else:
                        erfarg_low = i-binpos
                        erfarg_up = i+(1-binpos)
                    value = 0.5*erf(2*c*erfarg_up)-0.5*erf(2*c*erfarg_low)
                        
                    # divide by smearing_norm
                    value /= self.smearing_norm1
                    value /= (4*np.pi*deltaR**2)/self.cutoff_surface_area1 * self.binwidth1 * N1 * N2  # take N1=N2 into account
                    #value *= f_cutoff
                    feature[0][newbin] += value 
        return feature[0]

    def get_singleGradient(self, atoms):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """

        feature = [None]*2
        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_types = self.atomic_types
        atomic_count = self.atomic_count
        volume = self.volume

        nb_distVec, nb_deltaRs, nb_bondtype, np_index = self.__get_neighbour_lists(pos, num, pbc, cell, n_atoms)

        # Initialize fingerprint object
        feature = [{} for _ in range(2)]  # [None]*2

        # two body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                key = tuple(sorted([type1, type2]))
                if key not in feature[0]: 
                    feature[0][key] = np.zeros(self.Nbins1)

        # three body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                for type3 in self.atomic_types:
                    key = tuple([type1] + sorted([type1, type2]))
                    if key not in feature[0]:
                        feature[1][key] = np.zeros(self.Nbins2)

        for deltaR, dx, index in zip(R, dxMat, indexMat):
            rbin = int(np.floor(deltaR/self.binwidth))
            binpos = (deltaR % self.binwidth) / self.binwidth  # From 0 to binwidth (set constant at 0.5*binwidth for original)
            rabove = int(binpos > 0.5)

            # Lower and upper range of bins affected by the current atomic distance deltaR.
            minbin_lim = -self.m-(1-rabove)
            maxbin_lim = self.m+1+rabove
            for i in range(minbin_lim, maxbin_lim):
                newbin = rbin + i  # maybe abs() to make negative bins contribute aswell.
                if newbin < 0 or newbin >= self.Nbins:
                    continue

                c = 0.25*np.sqrt(2)*self.binwidth*1./self.sigma
                if i == minbin_lim:
                    arg_low = -(self.m+0.5)
                    arg_up = i+(1-binpos)
                elif i == maxbin_lim:
                    arg_low = i-binpos
                    arg_up = self.m+0.5
                else:
                    arg_low = i-binpos
                    arg_up = i+(1-binpos)
                value1 = -1./deltaR*(erf(2*c*arg_up)-erf(2*c*arg_low))
                value2 = -2*(np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))  # 2 in front..
                value = value1 + value2

                # divide by smearing_norm
                value /= self.smearing_norm
                value /= (4*np.pi*deltaR**2)/self.cutoffVolume * self.binwidth * N_distances

                # Add to the the gradient matrix
                fingerprint_grad[newbin, self.dim*index[0]:self.dim*index[0]+self.dim] += value/deltaR*dx
                fingerprint_grad[newbin, self.dim*index[1]:self.dim*index[1]+self.dim] += -value/deltaR*dx
        return fingerprint_grad
    
    def __get_neighbour_lists(self, pos, num, pbc, cell, n_atoms):

        # get_neighbourcells
        Rc_max = max(self.Rc1, 2*self.Rc2)  # relevant cutoff
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i])))
                neighbours.append(range(-ncellmax,ncellmax+1))
            else:
                neighbours.append([0])
        neighbourcells = []
        for x,y,z in product(*neighbours):
            neighbourcells.append((x,y,z))  # maybe: if norm(cell*[x,y,z]) < Rc_max: append

        # get_neighbour_lists
        neighbour_distVec = [[] for _ in range(n_atoms)]
        neighbour_deltaRs = [[] for _ in range(n_atoms)]
        neighbour_bondtype = [[] for _ in range(n_atoms)]
        neighbour_index = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = xyz @ cell
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1,self.dim)), displaced_pos)
                for j in range(n_atoms):
                    if deltaRs[j] < max(self.Rc1+self.nsigma*self.sigma1, self.Rc2) and deltaRs[j] > 1e-6:
                        neighbour_distVec[i].append(displaced)
                        neighbour_deltaRs[i].append(deltaRs[j])
                        neighbour_bondtype[i].append(tuple(sorted([num[i], num[j]])))
                        neighbour_index[i].append(j)

        return neighbour_distVec, neighbour_deltaRs, neighbour_bondtype, neighbour_index
