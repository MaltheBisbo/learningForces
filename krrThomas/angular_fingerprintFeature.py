import os
import sys
import numpy as np
from math import erf
from itertools import product

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
        self.atomic_count = [list(self.num).count(i) for i in self.atomic_types]
        self.volume = atoms.get_volume()

    def get_features(self,atoms,individual=False):
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
        neighbour_list = [{} for _ in range(n_atoms)]
        neighbour_list_exp = [{} for _ in range(n_atoms)]
        for i in range(n_atoms):
            for xyz in neighbourcells:
                displacement = np.dot(cell.T,np.array(xyz).T)
                displaced_pos = pos + displacement
                deltaRs = np.apply_along_axis(np.linalg.norm,1,displaced_pos-pos[i])
                for j in range(n_atoms):
                    if deltaRs[j] < self.Rc and deltaRs[j] > 1e-6:
                        neighbour_list[i][(xyz,j)] = deltaRs[j]
                    elif deltaRs[j] < 2*self.Rc and deltaRs[j] > 1e-6:
                        neighbour_list_exp[i][(xyz,j)] = deltaRs[j]

        # two component features
        feature[0] = [None]*n_atoms
        for i in range(n_atoms):
            feature[0][i] = {}
            for j in atomic_types:
                if (not sum(pbc)) and num[i] == j and atomic_count[atomic_types.index(j)] == 1:  # If i is the only atom of its type + no pbc
                    continue
                else:
                    bond_type = tuple(sorted([num[i],j]))
                    feature[0][i][bond_type] = []
            neighbours_i = neighbour_list[i].items()
            for j in range(len(neighbours_i)):
                neighbour_j = neighbours_i[j]
                _,index_j = neighbour_j[0]
                Rij = neighbour_j[1]
                bond_type = tuple(sorted([num[i],num[index_j]]))
                feature[0][i][bond_type].append(Rij)

        # three component features
        feature[1] = [None]*n_atoms
        for i in range(n_atoms):
            feature[1][i] = {}
            for j in atomic_types:
                if (not sum(pbc)) and  num[i] == j and atomic_count[atomic_types.index(j)] == 1:
                    continue
                else:
                    for k in atomic_types:
                        if (not sum(pbc)) and ((j == k and atomic_count[atomic_types.index(j)] == 1) or\
                                (num[i] == k and atomic_count[atomic_types.index(k)] == 1) or\
                                ((num[i] == j and j == k) and atomic_count[atomic_types.index(k)] < 3)):
                            continue
                        else:
                            bond_type = tuple([num[i]] + sorted([j,k]))
                            feature[1][i][bond_type] = []
            neighbours_i = neighbour_list[i].items()
            for j in range(len(neighbours_i)):
                neighbour_j = neighbours_i[j]
                shift_j,index_j = neighbour_j[0]
                Rij = neighbour_j[1]
                neighbours_j = neighbour_list[index_j]
                neighbours_j_exp = neighbour_list_exp[index_j]
                for k in range(j+1,len(neighbours_i)):
                    neighbour_k = neighbours_i[k]
                    shift_k,index_k = neighbour_k[0]
                    nxyz = (shift_k[0]-shift_j[0],shift_k[1]-shift_j[1],shift_k[2]-shift_j[2])
                    Rik = neighbour_k[1]
                    try:
                        Rjk = neighbours_j[(nxyz,index_k)]
                    except:
                        Rjk = neighbours_j_exp[(nxyz,index_k)]
                    lengths = [(num[i],num[index_j],Rij),(num[i],num[index_k],Rik),(num[index_j],num[index_k],Rjk)]
                    lengths = sorted(lengths[:2]) + [lengths[2]]
                    lengths = [lengths[0][2],lengths[1][2],lengths[2][2]]
                    bond_type = tuple([num[i]] + sorted([num[index_j],num[index_k]]))
                    feature[1][i][bond_type].append(lengths)


        # the fingerprint function
        fingerprints = {}
        type_combinations = []
        for i,type1 in enumerate(atomic_types):
            for type2 in atomic_types[i:]:
                type_combinations.append((type1,type2))

        # two component fingerprints
        nbins = int(np.ceil(self.Rc*1./self.binwidth1))
        m = int(np.ceil(self.nsigma*self.sigma1/self.binwidth1))
        smearing_norm = erf(0.25*np.sqrt(2)*self.binwidth1*(2*m+1)*1./self.sigma1)
        c = 0.25*np.sqrt(2)*self.binwidth1*1./self.sigma1
        
        for type1,type2 in type_combinations:
            key = (type1,type2)
            fingerprints[key] = np.zeros(nbins)
        for i in range(len(atoms)):
            type1 = num[i]
            for type2 in atomic_types:
                key = tuple(sorted([type1,type2]))
                fingerprints[key] += self._get_atomic_rdf(feature[0][i][key], nbins, m, smearing_norm, c)
        for type1,type2 in type_combinations:
            key = (type1,type2)
            fingerprints[key] /= 2*self.binwidth1*atomic_count[atomic_types.index(type1)]\
                                 *atomic_count[atomic_types.index(type2)]/volume
            fingerprints[key] -= 1

        # three component fingerprints
        nbins = int(np.ceil(np.pi/self.binwidth2))
        m = int(np.ceil(self.nsigma*self.sigma2/self.binwidth2))
        smearing_norm = erf(0.25*np.sqrt(2)*self.binwidth2*(2*m+1)*1./self.sigma2)
        c = 0.25*np.sqrt(2)*self.binwidth2*1./self.sigma2
        
        for type1 in atomic_types:
            for type2,type3 in type_combinations:
                key = (type1,type2,type3)
                fingerprints[key] = np.zeros(nbins)
        for i in range(len(atoms)):
            type1 = num[i]
            for type2,type3 in type_combinations:
                key = (type1,type2,type3)
                fingerprints[key] += self._get_atomic_adf(feature[1][i][key], nbins, m, smearing_norm, c)
        for type1 in atomic_types:
            for type2,type3 in type_combinations:
                key = (type1,type2,type3)
                fingerprints[key] /= self.binwidth2*atomic_count[atomic_types.index(type1)]\
                                     *atomic_count[atomic_types.index(type2)]*atomic_count[atomic_types.index(type3)]/volume
                fingerprints[key] -= 1

        return fingerprints

    def _get_atomic_rdf(self, bonds, nbins, m, smearing_norm, c):
        """ Helper function for get_features
        """
        rdf = np.zeros(nbins)
        for r in bonds:
            rbin = int(np.floor(r/self.binwidth1))
            for i in range(-m,m+1):
                newbin = rbin + i
                if newbin >= 0 and newbin < nbins:
                    value = 0.5*erf(c*(2*i+1))-0.5*erf(c*(2*i-1))
                    value /= smearing_norm
                    area = self._surface_area(r)
                    value /= area
                    rdf[newbin] += value
        return rdf

    def _get_atomic_adf(self, bonds, nbins, m, smearing_norm, c):
        """
        """
        adf = np.zeros(nbins)
        for tri_bond in bonds:
            a = self._angle(tri_bond)
            abin = int(np.floor(a/self.binwidth2))
            for i in range(-m,m+1):
                newbin = abin + i
                if newbin >= 0 and newbin < nbins:
                    value = 0.5*erf(c*(2*i+1))-0.5*erf(c*(2*i-1))
                    value /= smearing_norm
                    area1 = self._surface_area(tri_bond[0])
                    area2 = self._surface_area(tri_bond[1])
                    value /= area1 + area2
                    adf[newbin] += value
        return adf

    def _surface_area(self, r):
        """
        """
        return 4*np.pi*(r**2)

    def _angle(self, bond_set):
        a = bond_set[0]
        b = bond_set[1]
        c = bond_set[2]
        x = (a**2-b**2-c**2)/(2.*a*b)
        # this is added to correct for numerical errors
        if x < -1:
            x = -1.
        elif x > 1:
            x = 1.
        return np.arccos(x)

    def get_similarity(self, fp1, fp2):
        """
        """
        if isinstance(fp1,np.ndarray):
            fp1 = fp1[0]

        atomic_types = self.atomic_types
        atomic_count = self.atomic_count

        keys = fp1.keys()
        keys1 = sorted([k for k in keys if len(k) == 2])
        keys2 = sorted([k for k in keys if len(k) == 3])
        keys = keys1 + keys2

        # calculating the weights
        w = {}
        wtot1 = 0
        nw1 = len(keys1)
        for key in keys1:
            weight = atomic_count[atomic_types.index(key[0])]*atomic_count[atomic_types.index(key[1])]
            wtot1 += weight
            w[key] = weight
        for key in keys1:
            w[key] /= float(wtot1*nw1)
        wtot2 = 0
        nw2 = len(keys2)
        for key in keys2:
            weight = atomic_count[atomic_types.index(key[0])]*atomic_count[atomic_types.index(key[1])]\
                     *atomic_count[atomic_types.index(key[2])]
            wtot2 += weight
            w[key] = weight
        for key in keys2:
            w[key] /= float(wtot2*nw2)

        # calculating the fingerprint norms
        norm1 = 0
        norm2 = 0
        for key in keys:
            norm1 += np.linalg.norm(fp1[key])**2*w[key]
            norm2 += np.linalg.norm(fp2[key])**2*w[key]
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)

        # calculating the distance
        distance = 0
        for key in keys:
            distance += np.sum(fp1[key]*fp2[key])*w[key]/(norm1*norm2)

        distance = 0.5*(1-distance)

        return distance
