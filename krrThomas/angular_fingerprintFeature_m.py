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

global_count1 = 0

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
        if sum(self.pbc) != 0:
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
        self.cutoff_surface_area1 = 4*np.pi*self.Rc1**2
        self.cutoff_surface_area2 = 4*np.pi*self.Rc2**2

        # Elements in feature
        Nbondtypes_2body = 0
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                if type1 < type2:
                    Nbondtypes_2body += 1
                elif type1 == type2:  # and self.atomic_count[type1] > 1:
                    Nbondtypes_2body += 1
        Nelements_2body = self.Nbins1 * Nbondtypes_2body

        Nbondtypes_3body = 0
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                for type3 in self.atomic_types:
                    if type2 < type3:
                        Nbondtypes_3body += 1
                    elif type2 == type3:  # and self.atomic_count[type1] > 1:
                        Nbondtypes_3body += 1
        Nelements_3body = self.Nbins2 * Nbondtypes_3body
        
        self.Nelements = Nelements_2body + Nelements_3body
        
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
        if sum(pbc) != 0:
            volume = self.volume

        nb_distVec, nb_deltaRs, nb_bondtype, nb_index, nb_distVec_ang, nb_deltaRs_ang, nb_bondtype_ang, nb_index_ang = self.__get_neighbour_lists(pos, num, pbc, cell, n_atoms)

        # Initialize fingerprint object
        feature = [{} for _ in range(2)]  # [None]*2

        # two body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                key = tuple(sorted([type1, type2]))
                if key not in feature[0]: 
                    feature[0][key] = np.zeros(self.Nbins1)

        keys_2body = feature[0].keys()
        
        # three body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                for type3 in self.atomic_types:
                    key = tuple([type1] + sorted([type2, type3]))
                    if key not in feature[1]:
                        feature[1][key] = np.zeros(self.Nbins2)

        keys_3body = feature[1].keys()

        # Count the number of interacting atom-pairs
        N_distances = sum([len(x) for x in nb_deltaRs])
        
        # Two body
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs[j])):
                deltaR = nb_deltaRs[j][n]

                # use only distances relevant for 2body part
                if deltaR > self.Rc1 + self.nsigma*self.sigma1:
                    continue
                global global_count1
                global_count1 += 1
                
                # Identify what bin 'deltaR' belongs to + it's position in this bin
                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = (deltaR % self.binwidth1) / self.binwidth1  # From 0 to binwidth (set constant at 0.5*binwidth for original)

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                above_bin_center = int(binpos > 0.5)
                minbin_lim = -self.m1 - (1-above_bin_center)
                maxbin_lim = self.m1 + above_bin_center
                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue
                
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
                    type1, type2 = nb_bondtype[j][n]
                    value /= self.__surface_area(deltaR)/self.cutoff_surface_area1 * self.binwidth1 * \
                             atomic_count[type1] * atomic_count[type2]
                    
                    feature[0][nb_bondtype[j][n]][newbin] += value

        #if not self.use_angular:
        #    keys_2body = sorted(feature[0].keys())
        #    Nelements = len(keys_2body) * self.Nbins1
        #    fingerprint = np.zeros(Nelements)
        #    for i, key in enumerate(keys_2body):
        #        fingerprint[i*self.Nbins1 : (i+1)*self.Nbins1] = feature[0][key]
        #    #print('count1:', global_count1)
        #    return fingerprint
        
        # Three body
        for j in range(n_atoms):
            for n in range(len(nb_deltaRs_ang[j])):
                for m in range(n+1, len(nb_deltaRs_ang[j])):
                    type_j, type_n, type_m = nb_bondtype_ang[j][n][0], nb_bondtype_ang[j][n][1], nb_bondtype_ang[j][m][1]
                    bondtype_3body = tuple([type_j] + sorted([type_n, type_m]))

                    deltaR_n, deltaR_m = nb_deltaRs_ang[j][n], nb_deltaRs_ang[j][m]
                    if deltaR_n > self.Rc2 or deltaR_m > self.Rc2:
                        continue
                    angle = self.__angle(nb_distVec_ang[j][n], nb_distVec_ang[j][m])
                    center_bin = int(np.floor(angle/self.binwidth2))
                    binpos = (angle % self.binwidth2) / self.binwidth2  # From 0 to binwidth (set constant at 0.5*binwidth for original)
                    
                    #print('center bin:', center_bin/self.Nbins2*180)
                    #print('binpos:', binpos/self.Nbins2*180)
                    
                    
                    # Lower and upper range of bins affected by the current atomic distance deltaR.
                    above_bin_center = int(binpos > 0.5)
                    minbin_lim = -self.m2 - (1-above_bin_center)
                    maxbin_lim = self.m2 + above_bin_center
                    for i in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + i
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2-1:
                            newbin = self.Nbins2 - newbin % (self.Nbins2 - 1)
                        #print((newbin)/self.Nbins2*180)
                        #print(newbin)
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
                        # CHECK: considder if deltaR part is relevant
                        value /= (4*np.pi*(deltaR_n**2 + deltaR_m**2))/self.cutoff_surface_area2 * self.binwidth2 * \
                                 atomic_count[type_j] * atomic_count[type_n] * atomic_count[type_m]
                        value *= self.__f_cutoff(deltaR_n, self.gamma, self.Rc2)*self.__f_cutoff(deltaR_m, self.gamma, self.Rc2)
                        feature[1][bondtype_3body][newbin] += value

        keys_2body = sorted(feature[0].keys())
        keys_3body = sorted(feature[1].keys())
        Nelements = len(keys_2body) * self.Nbins1 + len(keys_3body) * self.Nbins2
        fingerprint = np.zeros(Nelements)
        for i, key in enumerate(keys_2body):
            fingerprint[i*self.Nbins1 : (i+1)*self.Nbins1] = feature[0][key]
        for i, key in enumerate(keys_3body):
            fingerprint[i*self.Nbins2 + len(keys_2body) * self.Nbins1 : (i+1)*self.Nbins2 + len(keys_2body) * self.Nbins1] = self.eta * feature[1][key]
        
        return fingerprint

    def get_featureGradients(self, atoms):
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
        if sum(pbc) != 0:
            volume = self.volume

        nb_distVec, nb_deltaRs, nb_bondtype, nb_index, nb_distVec_ang, nb_deltaRs_ang, nb_bondtype_ang, nb_index_ang = self.__get_neighbour_lists(pos, num, pbc, cell, n_atoms)

        # Initialize fingerprint object
        feature_grad = [{} for _ in range(2)]  # [None]*2

        # two body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                key = tuple(sorted([type1, type2]))
                if key not in feature_grad[0]: 
                    feature_grad[0][key] = np.zeros((self.Nbins1, n_atoms*self.dim))

        # three body
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                for type3 in self.atomic_types:
                    key = tuple([type1] + sorted([type1, type2]))
                    if key not in feature_grad[1]:
                        feature_grad[1][key] = np.zeros((self.Nbins2, n_atoms*self.dim))

        # Count the number of interacting atom-pairs
        N_distances = sum([len(x) for x in nb_deltaRs])

        for j in range(n_atoms):
            for n in range(len(nb_deltaRs[j])):
                deltaR = nb_deltaRs[j][n]
                if deltaR > self.Rc1:
                    continue
                dx = nb_distVec[j][n]
                index = nb_index[j][n]

                center_bin = int(np.floor(deltaR/self.binwidth1))
                binpos = (deltaR % self.binwidth1) / self.binwidth1  # From 0 to binwidth (set constant at 0.5*binwidth for original)
                above_bin_center = int(binpos > 0.5)
                
                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = -self.m1 - (1-above_bin_center)
                maxbin_lim = self.m1 + above_bin_center
                for i in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + i  # maybe abs() to make negative bins contribute aswell.
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    c = 0.25*np.sqrt(2)*self.binwidth1*1./self.sigma1
                    if i == minbin_lim:
                        arg_low = -(self.m1+0.5)
                        arg_up = i+(1-binpos)
                    elif i == maxbin_lim:
                        arg_low = i-binpos
                        arg_up = self.m1+0.5
                    else:
                        arg_low = i-binpos
                        arg_up = i+(1-binpos)
                    value1 = -1./deltaR*(erf(2*c*arg_up)-erf(2*c*arg_low))
                    value2 = -2*(np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))  # 2 in front..
                    value = value1 + value2
                    
                    # divide by smearing_norm
                    value /= self.smearing_norm1
                    value /= (4*np.pi*deltaR**2)/self.cutoff_surface_area1 * self.binwidth1 * N_distances

                    # Add to the the gradient matrix
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[0]:self.dim*index[0]+self.dim] += -value/deltaR*dx
                    feature_grad[0][nb_bondtype[j][n]][newbin, self.dim*index[1]:self.dim*index[1]+self.dim] += value/deltaR*dx
        return feature_grad[0]
    
    def __get_neighbour_lists(self, pos, num, pbc, cell, n_atoms):

        # get_neighbourcells
        Rc_max = max(self.Rc1+self.sigma1*self.nsigma, self.Rc2)  # relevant cutoff
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i]))) + 1 # +1 because atoms can be outside unitcell.
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

        neighbour_distVec_ang = [[] for _ in range(n_atoms)]
        neighbour_deltaRs_ang = [[] for _ in range(n_atoms)]
        neighbour_bondtype_ang = [[] for _ in range(n_atoms)]
        neighbour_index_ang = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                cell_displacement = xyz @ cell
                distVec = pos + cell_displacement
                deltaRs = cdist(pos[i].reshape((1, self.dim)), distVec).reshape(-1)
                for j in range(n_atoms):
                    if deltaRs[j] < max(self.Rc1+self.nsigma*self.sigma1, self.Rc2) and deltaRs[j] > 1e-6:
                        if j > i:
                            neighbour_distVec[i].append(distVec[j] - pos[i])
                            neighbour_deltaRs[i].append(deltaRs[j])
                            neighbour_bondtype[i].append(tuple(sorted([num[i], num[j]])))
                            neighbour_index[i].append((i,j))
                            
                            neighbour_distVec_ang[i].append(distVec[j] - pos[i])
                            neighbour_deltaRs_ang[i].append(deltaRs[j])
                            neighbour_bondtype_ang[i].append([num[i], num[j]])
                            neighbour_index_ang[i].append((i,j))
                            

        return neighbour_distVec, neighbour_deltaRs, neighbour_bondtype, neighbour_index, neighbour_distVec_ang, neighbour_deltaRs_ang, neighbour_bondtype_ang, neighbour_index_ang

    def __surface_area(self, r):
        """
        """
        return 4*np.pi*(r**2)
    
    def __angle(self, vec1, vec2):
        """
        Returns angle with convention [0,pi]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return np.arccos(np.dot(vec1,vec2)/(norm1*norm2))

    def __f_cutoff(self, r, gamma, Rc):
        """
        Polinomial cutoff function in the, with the steepness determined by "gamma"
        gamma = 2 resembels the cosine cutoff function.
        For large gamma, the function goes towards a step function at Rc.
        """
        return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma
