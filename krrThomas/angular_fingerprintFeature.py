from __future__ import print_function
import os
import sys
import numpy as np
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

import pdb

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

class Angular_Fingerprint(object):
    """ comparator for construction of angular fingerprints
    """

    def __init__(self, atoms, Rc1=4.0, Rc2=4.0, binwidth1=0.1, Nbins2=30, sigma1=0.2, sigma2=0.10, nsigma=4, eta=1, gamma=3, use_angular=True):
        """ Set a common cut of radius
        """
        self.Rc1 = Rc1
        self.Rc2 = Rc2
        self.binwidth1 = binwidth1
        self.Nbins2 = Nbins2
        self.binwidth2 = np.pi / Nbins2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.nsigma = nsigma
        self.eta = eta
        self.gamma = gamma
        self.use_angular = use_angular

        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        self.n_atoms = atoms.get_number_of_atoms()
        self.num = atoms.get_atomic_numbers()
        self.atomic_types = sorted(list(set(self.num)))
        self.atomic_count = {type:list(self.num).count(type) for type in self.atomic_types}
        self.volume = atoms.get_volume()
        self.dim = 3

        # parameters for the binning:
        self.m1 = self.nsigma*self.sigma1/self.binwidth1  # number of neighbour bins included.
        self.smearing_norm1 = erf(1/np.sqrt(2) * self.m1 * self.binwidth1/self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = self.nsigma*self.sigma2/self.binwidth2  # number of neighbour bins included.
        self.smearing_norm2 = erf(1/np.sqrt(2) * self.m2 * self.binwidth2/self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2

        # Create bondtype lists
        # 2body
        self.bondtypes_2body = []
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                key = tuple(sorted([type1, type2]))
                if type1 < type2:
                    self.bondtypes_2body.append(key)
                elif type1 == type2 and (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                    self.bondtypes_2body.append(key)
        Nelements_2body = self.Nbins1 * len(self.bondtypes_2body)

        # 3body
        self.bondtypes_3body = []
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                if type1 == type2 and not (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                    continue
                for type3 in self.atomic_types:
                    if type1 == type3 and not (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                        continue
                    key = tuple([type1] + sorted([type2, type3]))
                    if type2 < type3:
                        self.bondtypes_3body.append(key)
                    elif type2 == type3:
                        if type2 == type1 and (self.atomic_count[type1] > 2 or sum(self.pbc) > 0):
                            self.bondtypes_3body.append(key)
                        elif type2 != type1 and (self.atomic_count[type2] > 1 or sum(self.pbc) > 0):
                            self.bondtypes_3body.append(key)
        Nelements_3body = self.Nbins2 * len(self.bondtypes_3body)

        if use_angular:
            self.Nelements = Nelements_2body + Nelements_3body
        else:
            self.Nelements = Nelements_2body

    def get_feature(self, atoms):
        """
        """

        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_count = self.atomic_count

        # Initialize fingerprint
        feature = [{} for _ in range(2)]

        # Get relevant neighbour unit-cells
        neighbourcells = self.__get_neighbour_cells(pbc, cell)

        ## Radial part ##

        # Calculate neighbour lists - 2body
        nb_deltaRs = [[] for _ in range(n_atoms)]
        nb_bondtype = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = np.dot(xyz, cell)
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < self.Rc1+self.nsigma*self.sigma1 and deltaRs[j] > 1e-6:
                        if j >= 0:
                            nb_deltaRs[i].append(deltaRs[j])
                            nb_bondtype[i].append(tuple(sorted([num[i], num[j]])))

        # Initialize 2body bondtype dictionary
        for bondtype in self.bondtypes_2body:
            feature[0][bondtype] = np.zeros(self.Nbins1)

        # Calculate radial features
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs[j])):
                deltaR = nb_deltaRs[j][n]

                # Calculate normalization
                type1, type2 = nb_bondtype[j][n]
                num_pairs = atomic_count[type1] * atomic_count[type2]
                normalization = 1./self.smearing_norm1
                normalization /= 4*np.pi*deltaR**2 * self.binwidth1 * num_pairs/self.volume
                #value *= self.__f_cutoff(deltaR, self.gamma, self.Rc1)

                # Identify what bin 'deltaR' belongs to + it's position in this bin
                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = deltaR/self.binwidth1 - center_bin

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = -int(np.ceil(self.m1 - binpos))
                maxbin_lim = int(np.ceil(self.m1 - (1-binpos)))

                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 1./np.sqrt(2)*self.binwidth1/self.sigma1
                    erfarg_low = max(-self.m1, i-binpos)
                    erfarg_up = min(self.m1, i+(1-binpos))
                    value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)

                    # Apply normalization
                    value *= normalization

                    feature[0][nb_bondtype[j][n]][newbin] += value

        # Return feature - if angular part is not required
        if not self.use_angular:
            fingerprint = np.zeros(self.Nelements)
            for i, key in enumerate(self.bondtypes_2body):
                fingerprint[i*self.Nbins1: (i+1)*self.Nbins1] = feature[0][key]
            return fingerprint

        ## Angular part ##

        # Calculate neighbour lists - 3body
        nb_deltaRs_ang = [[] for _ in range(n_atoms)]
        nb_bondtype_ang = [[] for _ in range(n_atoms)]
        nb_distVec_ang = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = np.dot(xyz, cell)
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < self.Rc2 and deltaRs[j] > 1e-6:
                        if j >= 0:
                            nb_deltaRs_ang[i].append(deltaRs[j])
                            nb_bondtype_ang[i].append(tuple([num[i], num[j]]))
                            nb_distVec_ang[i].append(distVec[j] - pos[i])

        # Initialize 3body bondtype dictionary
        for bondtype in self.bondtypes_3body:
            feature[1][bondtype] = np.zeros(self.Nbins2)

        # Calculate angular features
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs_ang[j])):
                for m in range(n+1, len(nb_deltaRs_ang[j])):
                    type_j, type_n, type_m = nb_bondtype_ang[j][n][0], nb_bondtype_ang[j][n][1], nb_bondtype_ang[j][m][1]
                    deltaR_n, deltaR_m = nb_deltaRs_ang[j][n], nb_deltaRs_ang[j][m]

                    # Use only distances relevant for 3body part
                    if deltaR_n > self.Rc2 or deltaR_m > self.Rc2:
                        continue

                    bondtype = tuple([type_j] + sorted([type_n, type_m]))
                    angle, _ = self.__angle(nb_distVec_ang[j][n], nb_distVec_ang[j][m])

                    # Calculate normalization
                    num_pairs = atomic_count[type_j] * atomic_count[type_n] * atomic_count[type_m]
                    normalization = 1./self.smearing_norm2
                    normalization /= num_pairs/self.volume

                    # Identify what bin 'angle' belongs to + it's position in this bin
                    center_bin = int(np.floor(angle/self.binwidth2))
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current angle.
                    minbin_lim = -int(np.ceil(self.m2 - binpos))
                    maxbin_lim = int(np.ceil(self.m2 - (1-binpos)))
                    for i in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + i

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2-1:
                            newbin = 2*self.Nbins2 - newbin - 1

                        # Calculate gauss+cutoff contribution to current bin
                        c = 1./np.sqrt(2)*self.binwidth2/self.sigma2
                        erfarg_low = max(-self.m2, i-binpos)
                        erfarg_up = min(self.m2, i+(1-binpos))
                        value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)
                        value *= self.__f_cutoff(deltaR_n, self.gamma, self.Rc2)
                        value *= self.__f_cutoff(deltaR_m, self.gamma, self.Rc2)

                        # Apply normalization
                        value *= normalization

                        feature[1][bondtype][newbin] += value

        fingerprint = np.zeros(self.Nelements)
        for i, key in enumerate(self.bondtypes_2body):
            fingerprint[i*self.Nbins1: (i+1)*self.Nbins1] = feature[0][key]
        Nelements_2body = self.Nbins1 * len(self.bondtypes_2body)
        for i, key in enumerate(self.bondtypes_3body):
            fingerprint[i*self.Nbins2 + Nelements_2body: (i+1)*self.Nbins2 + Nelements_2body] = self.eta * feature[1][key]
        return fingerprint

    def get_featureMat(self, atoms_list, show_progress=False):
        if show_progress:
            featureMat = []
            for i, atoms in enumerate(atoms_list):
                print('Calculating FeatureMat {}/{}\r'.format(i, len(atoms_list)), end='')
                featureMat.append(self.get_feature(atoms))
            print('\n')
        else:
            featureMat = np.array([self.get_feature(atoms) for atoms in atoms_list])
        featureMat = np.array(featureMat)
        return featureMat

    def get_featureGradient(self, atoms):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """

        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_count = self.atomic_count

        # Get relevant neighbour unit-cells
        neighbourcells = self.__get_neighbour_cells(pbc, cell)

        # Calculate neighbour lists
        nb_distVec = [[] for _ in range(n_atoms)]
        nb_deltaRs = [[] for _ in range(n_atoms)]
        nb_bondtype = [[] for _ in range(n_atoms)]
        nb_index = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = np.dot(xyz, cell)
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < max(self.Rc1+self.nsigma*self.sigma1, self.Rc2) and deltaRs[j] > 1e-6:
                        if j >= 0:
                            nb_distVec[i].append(distVec[j] - pos[i])
                            nb_deltaRs[i].append(deltaRs[j])
                            nb_bondtype[i].append(tuple(sorted([num[i], num[j]])))
                            nb_index[i].append((i,j))

        # Initialize fingerprint gradient dictionaries
        feature_grad = [{} for _ in range(2)]

        ## Radial part ##

        # Initialize 2body bondtype dictionary
        for bondtype in self.bondtypes_2body:
            feature_grad[0][bondtype] = np.zeros((self.Nbins1, n_atoms*self.dim))

        # Radial feature gradient
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs[j])):
                deltaR = nb_deltaRs[j][n]

                # use only distances relevant for 2body part
                if deltaR > self.Rc1 + self.nsigma*self.sigma1:
                    continue
                dx = nb_distVec[j][n]
                index = nb_index[j][n]

                # Calculate normalization
                type1, type2 = nb_bondtype[j][n]
                num_pairs = atomic_count[type1] * atomic_count[type2]
                normalization = 1/self.smearing_norm1
                normalization /= 4*np.pi*deltaR**2 * self.binwidth1 * num_pairs/self.volume

                # Identify what bin 'deltaR' belongs to + it's position in this bin
                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = deltaR/self.binwidth1 - center_bin

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = -int(np.ceil(self.m1 - binpos))
                maxbin_lim = int(np.ceil(self.m1 - (1-binpos)))
                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 1/np.sqrt(2)*self.binwidth1/self.sigma1
                    arg_low = max(-self.m1, i-binpos)
                    arg_up = min(self.m1, i+(1-binpos))
                    value1 = -1./deltaR*(erf(c*arg_up)-erf(c*arg_low))
                    value2 = -1/(self.sigma1*np.sqrt(2*np.pi)) * (np.exp(-(c*arg_up)**2) - np.exp(-(c*arg_low)**2))
                    value = value1 + value2

                    # Apply normalization
                    value *= normalization

                    # Add to the the gradient matrix
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[0]:self.dim*index[0]+self.dim] += -value/deltaR*dx
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[1]:self.dim*index[1]+self.dim] += value/deltaR*dx

        # Return feature - if angular part is not required
        if not self.use_angular:
            fingerprint_grad = np.zeros((n_atoms*self.dim, self.Nelements))
            for i, key in enumerate(self.bondtypes_2body):
                fingerprint_grad[:, i*self.Nbins1: (i+1)*self.Nbins1] = feature_grad[0][key].T
            return fingerprint_grad

        ## Angular part ##

        # Calculate neighbour lists - 3body
        nb_deltaRs_ang = [[] for _ in range(n_atoms)]
        nb_bondtype_ang = [[] for _ in range(n_atoms)]
        nb_distVec_ang = [[] for _ in range(n_atoms)]
        nb_index_ang = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = np.dot(xyz, cell)
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < self.Rc2 and deltaRs[j] > 1e-6:
                        if j >= 0:
                            nb_deltaRs_ang[i].append(deltaRs[j])
                            nb_bondtype_ang[i].append(tuple([num[i], num[j]]))
                            nb_distVec_ang[i].append(distVec[j] - pos[i])
                            nb_index_ang[i].append((i,j))

        # Initialize 3body bondtype dictionary
        for bondtype in self.bondtypes_3body:
            feature_grad[1][bondtype] = np.zeros((self.Nbins2, n_atoms*self.dim))

        # Calculate angular features
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs_ang[j])):
                for m in range(n+1, len(nb_deltaRs_ang[j])):
                    type_j, type_n, type_m = nb_bondtype_ang[j][n][0], nb_bondtype_ang[j][n][1], nb_bondtype_ang[j][m][1]
                    deltaR_n, deltaR_m = nb_deltaRs_ang[j][n], nb_deltaRs_ang[j][m]

                    # Use only distances relevant for 3body part
                    if deltaR_n > self.Rc2 or deltaR_m > self.Rc2:
                        continue

                    bondtype = tuple([type_j] + sorted([type_n, type_m]))
                    angle, cos_angle = self.__angle(nb_distVec_ang[j][n], nb_distVec_ang[j][m])

                    # Calculate normalization
                    num_pairs = atomic_count[type_j] * atomic_count[type_n] * atomic_count[type_m]
                    normalization = 1./self.smearing_norm2
                    normalization /= num_pairs/self.volume

                    # Identify what bin 'angle' belongs to + it's position in this bin
                    center_bin = int(np.floor(angle/self.binwidth2))
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current angle.
                    minbin_lim = -int(np.ceil(self.m2 - binpos))
                    maxbin_lim = int(np.ceil(self.m2 - (1-binpos)))
                    for i in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + i

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2 - 1:
                            newbin = 2*self.Nbins2 - newbin - 1

                        # Calculate gauss contribution to current bin
                        c = 1./np.sqrt(2)*self.binwidth2 / self.sigma2
                        arg_low = max(-self.m2, i-binpos)
                        arg_up = min(self.m2, i+(1-binpos))
                        value1 = 0.5*erf(c*arg_up)-0.5*erf(c*arg_low)
                        value2 = -1./(self.sigma2*np.sqrt(2*np.pi)) * (np.exp(-(c*arg_up)**2) - np.exp(-(c*arg_low)**2))

                        # Apply normalization
                        value1 *= normalization
                        value2 *= normalization

                        fc_jn = self.__f_cutoff(deltaR_n, self.gamma, self.Rc2)
                        fc_jm = self.__f_cutoff(deltaR_m, self.gamma, self.Rc2)
                        fc_jn_grad = self.__f_cutoff_grad(deltaR_n, self.gamma, self.Rc2)
                        fc_jm_grad = self.__f_cutoff_grad(deltaR_m, self.gamma, self.Rc2)

                        dx_jn = nb_distVec_ang[j][n]
                        dx_jm = nb_distVec_ang[j][m]
                        index_jn = nb_index_ang[j][n]
                        index_jm = nb_index_ang[j][m]

                        if not (angle == 0 or angle == np.pi):
                            a = -1/np.sqrt(1 - cos_angle**2)
                            angle_n_grad = a * (dx_jm/(deltaR_n*deltaR_m) - cos_angle*dx_jn/deltaR_n**2)
                            angle_m_grad = a * (dx_jn/(deltaR_n*deltaR_m) - cos_angle*dx_jm/deltaR_m**2)
                            angle_j_grad = -(angle_n_grad + angle_m_grad)
                        else:
                            angle_j_grad, angle_n_grad, angle_m_grad = (0,0,0)

                        # Define the index range of the gradient that belongs to each atom
                        index_range_j = np.arange(self.dim*index_jn[0], self.dim*index_jn[0]+self.dim)
                        index_range_n = np.arange(self.dim*index_jn[1], self.dim*index_jn[1]+self.dim)
                        index_range_m = np.arange(self.dim*index_jm[1], self.dim*index_jm[1]+self.dim)

                        # Add to the the gradient matrix
                        feature_grad[1][bondtype][newbin, index_range_j] += -value1 * fc_jm*fc_jn_grad * dx_jn/deltaR_n
                        feature_grad[1][bondtype][newbin, index_range_n] += value1 * fc_jm*fc_jn_grad * dx_jn/deltaR_n

                        feature_grad[1][bondtype][newbin, index_range_j] += -value1 * fc_jn*fc_jm_grad * dx_jm/deltaR_m
                        feature_grad[1][bondtype][newbin, index_range_m] += value1 * fc_jn*fc_jm_grad * dx_jm/deltaR_m

                        feature_grad[1][bondtype][newbin, index_range_j] += value2 * angle_j_grad * fc_jn * fc_jm
                        feature_grad[1][bondtype][newbin, index_range_n] += value2 * angle_n_grad * fc_jn * fc_jm
                        feature_grad[1][bondtype][newbin, index_range_m] += value2 * angle_m_grad * fc_jn * fc_jm

        fingerprint_grad = np.zeros((n_atoms*self.dim, self.Nelements))
        for i, key in enumerate(self.bondtypes_2body):
            fingerprint_grad[:, i*self.Nbins1: (i+1)*self.Nbins1] = feature_grad[0][key].T
        Nelements_2body = self.Nbins1 * len(self.bondtypes_2body)
        for i, key in enumerate(self.bondtypes_3body):
            fingerprint_grad[:, i*self.Nbins2 + Nelements_2body: (i+1)*self.Nbins2 + Nelements_2body] = self.eta * feature_grad[1][key].T
        return fingerprint_grad

    def get_all_featureGradients(self, atoms_list, show_progress=False):
        if show_progress:
            feature_grads = []
            for i, atoms in enumerate(atoms_list):
                print('Calculating Feature Gradients {}/{}\r'.format(i, len(atoms_list)), end='')
                feature_grads.append(self.get_featureGradient(atoms))
            print('\n')
        else:
            feature_grads = np.array([self.get_featureGradient(atoms) for atoms in atoms_list])
        feature_grads = np.array(feature_grads)
        return feature_grads

    def __get_neighbour_cells(self, pbc, cell):

        # Calculate neighbour cells
        Rc_max = max(self.Rc1+self.sigma1*self.nsigma, self.Rc2)  # relevant cutoff
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i]))) + 1  # +1 because atoms can be outside unitcell.
                neighbours.append(range(-ncellmax,ncellmax+1))
            else:
                neighbours.append([0])

        neighbourcells = []
        for x,y,z in product(*neighbours):
            neighbourcells.append((x,y,z))

        return neighbourcells

    def __angle(self, vec1, vec2):
        """
        Returns angle with convention [0,pi]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        arg = np.dot(vec1,vec2)/(norm1*norm2)
        # This is added to correct for numerical errors
        if arg < -1:
            arg = -1.
        elif arg > 1:
            arg = 1.
        return np.arccos(arg), arg

    def __f_cutoff(self, r, gamma, Rc):
        """
        Polinomial cutoff function in the, with the steepness determined by "gamma"
        gamma = 2 resembels the cosine cutoff function.
        For large gamma, the function goes towards a step function at Rc.
        """
        if not gamma == 0:
            return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma
        else:
            return 1

    def __f_cutoff_grad(self, r, gamma, Rc):
        if not gamma == 0:
            return gamma*(gamma+1)/Rc * ((r/Rc)**gamma - (r/Rc)**(gamma-1))
        else:
            return 0

    def angle2_grad(self, RijVec, RikVec):
        Rij = np.linalg.norm(RijVec)
        Rik = np.linalg.norm(RikVec)

        a = RijVec/Rij - RikVec/Rik
        b = RijVec/Rij + RikVec/Rik
        A = np.linalg.norm(a)
        B = np.linalg.norm(b)
        D = A/B

        RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
        RikMat = np.dot(RikVec[:,np.newaxis], RikVec[:,np.newaxis].T)

        a_grad_j = -1/Rij**3 * RijMat + 1/Rij * np.identity(3)
        b_grad_j = a_grad_j

        a_sum_j = np.sum(a*a_grad_j, axis=1)
        b_sum_j = np.sum(b*b_grad_j, axis=1)

        grad_j = 2/(1+D**2) * (1/(A*B) * a_sum_j - A/(B**3) * b_sum_j)



        a_grad_k = 1/Rik**3 * RikMat - 1/Rik * np.identity(3)
        b_grad_k = -a_grad_k

        a_sum_k = np.sum(a*a_grad_k, axis=1)
        b_sum_k = np.sum(b*b_grad_k, axis=1)

        grad_k = 2/(1+D**2) * (1/(A*B) * a_sum_k - A/(B**3) * b_sum_k)


        a_grad_i = -(a_grad_j + a_grad_k)
        b_grad_i = -(b_grad_j + b_grad_k)

        a_sum_i = np.sum(a*a_grad_i, axis=1)
        b_sum_i = np.sum(b*b_grad_i, axis=1)

        grad_i = 2/(1+D**2) * (1/(A*B) * a_sum_i - A/(B**3) * b_sum_i)

        return grad_i, grad_j, grad_k
