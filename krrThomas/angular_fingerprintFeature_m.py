import os
import sys
import numpy as np
from math import erf
from itertools import product
from scipy.spatial.distance import cdist
from ase.visualize import view

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
        self.m1 = int(np.ceil(self.nsigma*self.sigma1/self.binwidth1))  # number of neighbour bins included.
        self.smearing_norm1 = erf(0.25*np.sqrt(2)*self.binwidth1*(2*self.m1+1)*1./self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = int(np.ceil(self.nsigma*self.sigma2/self.binwidth2))  # number of neighbour bins included.
        self.smearing_norm2 = erf(0.25*np.sqrt(2)*self.binwidth2*(2*self.m2+1)*1./self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2
        
        # Cutoff surface areas
        self.cutoff_volume1 = 4/3*np.pi*self.Rc1**3
        self.cutoff_surface_area2 = 4*np.pi*self.Rc2**2

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
        
    def get_features(self, atoms):
        """
        """
        # Wrap atoms into unit-cell
        atoms.wrap()
        
        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_types = self.atomic_types
        atomic_count = self.atomic_count

        # Get relevant neighbour unit-cells
        neighbourcells = self.__get_neighbour_cells(pbc, cell)

        # Calculate neighbour lists - 2body
        nb_deltaRs = [[] for _ in range(n_atoms)]
        nb_bondtype = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = np.dot(xyz, cell)
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < max(self.Rc1+self.nsigma*self.sigma1, self.Rc2) and deltaRs[j] > 1e-6:
                        if j >= 0:
                            nb_deltaRs[i].append(deltaRs[j])
                            nb_bondtype[i].append(tuple(sorted([num[i], num[j]])))
        
        # Initialize fingerprint object
        feature = [{} for _ in range(2)]

        ## Radial part ##
        
        # Initialize 2body bondtype dictionary
        for bondtype in self.bondtypes_2body:
            feature[0][bondtype] = np.zeros(self.Nbins1)
            
        # Calculate radial features
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs[j])):
                deltaR = nb_deltaRs[j][n]

                # use only distances relevant for 2body part
                if deltaR > self.Rc1 + self.nsigma*self.sigma1:
                    continue

                # Identify what bin 'deltaR' belongs to + it's position in this bin
                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = deltaR/self.binwidth1 - center_bin  # From 0 to binwidth (set constant at 0.5*binwidth for original)

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                above_bin_center = int(binpos > 0.5)
                minbin_lim = -self.m1 - (1-above_bin_center)
                maxbin_lim = self.m1 + above_bin_center
                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 0.25*np.sqrt(2)*self.binwidth1*1./self.sigma1
                    if i == minbin_lim:
                        erfarg_low = -(self.m1+0.5)
                        erfarg_up = i+(1-binpos)
                    elif i == maxbin_lim:
                        erfarg_low = i-binpos
                        erfarg_up = self.m1+0.5
                    else:
                        erfarg_low = i-binpos
                        erfarg_up = i+(1-binpos)
                    value = 0.5*erf(2*c*erfarg_up)-0.5*erf(2*c*erfarg_low)
                    
                    # divide by smearing_norm
                    value /= self.smearing_norm1

                    # Normalize
                    type1, type2 = nb_bondtype[j][n]
                    num_pairs = atomic_count[type1] * atomic_count[type2]
                    value /= 4*np.pi*deltaR**2 * self.binwidth1 * num_pairs/self.volume
                    #value *= self.__f_cutoff(deltaR, self.gamma, self.Rc1)
                    
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
                    
                    # Identify what bin 'angle' belongs to + it's position in this bin
                    center_bin = int(np.floor(angle/self.binwidth2))
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current angle.
                    above_bin_center = int(binpos > 0.5)
                    minbin_lim = -self.m2 - (1-above_bin_center)
                    maxbin_lim = self.m2 + above_bin_center
                    for i in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + i

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2-1:
                            newbin = self.Nbins2 - newbin % (self.Nbins2 - 1)

                        # Calculate gauss contribution to current bin
                        c = 0.25*np.sqrt(2)*self.binwidth2*1./self.sigma2
                        if i == minbin_lim:
                            erfarg_low = -(self.m2+0.5)
                            erfarg_up = i+(1-binpos)
                        elif i == maxbin_lim:
                            erfarg_low = i-binpos
                            erfarg_up = self.m2+0.5
                        else:
                            erfarg_low = i-binpos
                            erfarg_up = i+(1-binpos)
                        value = 0.5*erf(2*c*erfarg_up)-0.5*erf(2*c*erfarg_low)
                        
                        # divide by smearing_norm
                        value /= self.smearing_norm2

                        num_pairs = atomic_count[type_j] * atomic_count[type_n] * atomic_count[type_m]
                        #bin_volume = 2/3*np.pi*self.Rc2**3*(np.cos(angle-self.binwidth2/2) - np.cos(angle+self.binwidth2/2))
                        value /= num_pairs/self.volume
                        #value /= bin_volume * num_pairs/self.volume
                        #value /= (4*np.pi*(deltaR_n**2 + deltaR_m**2)) * self.binwidth2 * num_pairs/self.volume
                        value *= self.__f_cutoff(deltaR_n, self.gamma, self.Rc2)*self.__f_cutoff(deltaR_m, self.gamma, self.Rc2)

                        feature[1][bondtype][newbin] += value
        
        fingerprint = np.zeros(self.Nelements)
        for i, key in enumerate(self.bondtypes_2body):
            fingerprint[i*self.Nbins1: (i+1)*self.Nbins1] = feature[0][key]
        for i, key in enumerate(self.bondtypes_3body):
            fingerprint[i*self.Nbins2 + len(self.bondtypes_2body) * self.Nbins1: (i+1)*self.Nbins2 + len(self.bondtypes_2body) * self.Nbins1] = self.eta * feature[1][key]
        
        return fingerprint

    def get_featureGradients(self, atoms):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """

        # Wrap atoms into unit-cell
        atoms.wrap()
        
        pbc = self.pbc
        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        atomic_types = self.atomic_types
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

                # Identify what bin 'deltaR' belongs to + it's position in this bin
                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = deltaR/self.binwidth1 - center_bin  # From 0 to binwidth (set constant at 0.5*binwidth for original)

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                above_bin_center = int(binpos > 0.5)
                minbin_lim = -self.m1 - (1-above_bin_center)
                maxbin_lim = self.m1 + above_bin_center
                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    c = 0.25*np.sqrt(2)*self.binwidth1/self.sigma1
                    if i == minbin_lim:
                        arg_low = -(self.m1+0.5)
                        arg_up = i+(1-binpos)
                    elif i == maxbin_lim:
                        arg_low = i-binpos
                        arg_up = self.m1+0.5
                    else:
                        arg_low = i-binpos
                        arg_up = i+(1-binpos)
                    #fc = self.__f_cutoff(deltaR, self.gamma, self.Rc1)
                    #fc_grad = self.__f_cutoff_grad(deltaR, self.gamma, self.Rc1)
                    #value1 = (0.5*fc_grad - fc/deltaR)*(erf(2*c*arg_up)-erf(2*c*arg_low))
                    #value2 = -fc * 1/(self.sigma1*np.sqrt(2*np.pi)) * (np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))
                    value1 = -1./deltaR*(erf(2*c*arg_up)-erf(2*c*arg_low))
                    value2 = -1/(self.sigma1*np.sqrt(2*np.pi)) * (np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))
                    value = value1 + value2
                    
                    # divide by smearing_norm
                    value /= self.smearing_norm1

                    # Normalize
                    type1, type2 = nb_bondtype[j][n]
                    num_pairs = atomic_count[type1] * atomic_count[type2]
                    value /= 4*np.pi*deltaR**2 * self.binwidth1 * num_pairs/self.volume

                    # Add to the the gradient matrix
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[0]:self.dim*index[0]+self.dim] += -value/deltaR*dx
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[1]:self.dim*index[1]+self.dim] += value/deltaR*dx

        # Return feature - if angular part is not required
        if not self.use_angular:
            fingerprint_grad = np.zeros((self.Nelements, n_atoms*self.dim))
            for i, key in enumerate(self.bondtypes_2body):
                fingerprint_grad[i*self.Nbins1: (i+1)*self.Nbins1, :] = feature_grad[0][key]
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
                    
                    # Identify what bin 'angle' belongs to + it's position in this bin
                    center_bin = int(np.floor(angle/self.binwidth2))
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current angle.
                    above_bin_center = int(binpos > 0.5)
                    minbin_lim = -self.m2 - (1-above_bin_center)
                    maxbin_lim = self.m2 + above_bin_center
                    for i in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + i

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2 - 1:
                            newbin = self.Nbins2 - newbin % (self.Nbins2 - 1)

                        # Calculate gauss contribution to current bin
                        c = 0.25*np.sqrt(2)*self.binwidth2 / self.sigma2
                        if i == minbin_lim:
                            erfarg_low = -(self.m2+0.5)
                            erfarg_up = i+(1-binpos)
                        elif i == maxbin_lim:
                            erfarg_low = i-binpos
                            erfarg_up = self.m2+0.5
                        else:
                            erfarg_low = i-binpos
                            erfarg_up = i+(1-binpos)
                        value1 = 0.5*erf(2*c*erfarg_up)-0.5*erf(2*c*erfarg_low)
                        value2 = -1/(self.sigma1*np.sqrt(2*np.pi)) * (np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))

                        

                        
                        
                        # divide by smearing_norm
                        value1 /= self.smearing_norm2
                        value2 /= self.smearing_norm2

                        # Normalize
                        num_pairs = atomic_count[type_j] * atomic_count[type_n] * atomic_count[type_m]
                        value1 /= num_pairs/self.volume
                        value2 /= num_pairs/self.volume

                        
                        fc_jn = self.__f_cutoff(deltaR_n, self.gamma, self.Rc2)
                        fc_jm = self.__f_cutoff(deltaR_m, self.gamma, self.Rc2)
                        fc_jn_grad = self.__f_cutoff_grad(deltaR_n, self.gamma, self.Rc2)
                        fc_jm_grad = self.__f_cutoff_grad(deltaR_m, self.gamma, self.Rc2)

                        dx_jn = nb_distVec[j][n]
                        dx_jm = nb_distVec[j][m]
                        index_jn = nb_index[j][n]
                        index_jm = nb_index[j][m]

                        
                        a = -1/np.sqrt(1 - cos_angle**2)
                        angle_j_grad = a * ( - (dx_jn + dx_jm)/(deltaR_n*deltaR_m) + cos_angle*(dx_jn/deltaR_n**2 + dx_jm/deltaR_m**2) ) 
                        angle_n_grad = a * ( dx_jm/(deltaR_n*deltaR_m) - cos_angle*dx_jn/deltaR_n**2 )
                        angle_m_grad = a * ( dx_jn/(deltaR_n*deltaR_m) - cos_angle*dx_jm/deltaR_m**2 )
                        
                        # Define the index range of the gradient that belongs to each atom
                        index_range_j = np.arange(self.dim*index_jn[0]:self.dim*index_jn[0]+self.dim)
                        index_range_n = np.arange(self.dim*index_jn[1]:self.dim*index_jn[1]+self.dim)
                        index_range_m = np.arange(self.dim*index_jm[1]:self.dim*index_jm[1]+self.dim)
                        
                        # Add to the the gradient matrix
                        feature_grad[1][nb_bondtype[j][n]][newbin, index_range_j] += -value1 * fc_jm*fc_jn_grad * dx/deltaR_n
                        feature_grad[1][nb_bondtype[j][n]][newbin, index_range_n] += value1 * fc_jm*fc_jn_grad * dx/deltaR_n

                        feature_grad[1][nb_bondtype[j][m]][newbin, index_range_j] += -value1 * fc_jn*fc_jm_grad * dx/deltaR_m
                        feature_grad[1][nb_bondtype[j][m]][newbin, index_range_m] += value1 * fc_jn*fc_jm_grad * dx/deltaR_m

                        feature_grad[1][nb_bondtype[j][n]][newbin, index_range_j] += value2 * angle_j_grad * fc_jn * fc_jm
                        feature_grad[1][nb_bondtype[j][n]][newbin, index_range_n] += value2 * angle_n_grad * fc_jn * fc_jm
                        feature_grad[1][nb_bondtype[j][m]][newbin, index_range_m] += value2 * angle_m_grad * fc_jn * fc_jm

                        
        fingerprint = np.zeros(self.Nelements)
        for i, key in enumerate(self.bondtypes_2body):
            fingerprint[i*self.Nbins1: (i+1)*self.Nbins1] = feature[0][key]
        for i, key in enumerate(self.bondtypes_3body):
            fingerprint[i*self.Nbins2 + len(self.bondtypes_2body) * self.Nbins1: (i+1)*self.Nbins2 + len(self.bondtypes_2body) * self.Nbins1] = self.eta * feature[1][key]

        
    def __get_neighbour_cells(self, pbc, cell):

        # Calculate neighbour cells
        Rc_max = max(self.Rc1+self.sigma1*self.nsigma, self.Rc2)  # relevant cutoff
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i])))  # + 1  # +1 because atoms can be outside unitcell.
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
