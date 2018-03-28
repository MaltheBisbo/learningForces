import os
import sys
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

import time

import numpy as np
cimport numpy as np

from libc.math cimport *

from cymem.cymem cimport Pool
cimport cython

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

# Custom functions
ctypedef struct Point:
    double coord[3]

cdef Point subtract(Point p1, Point p2):
    cdef Point p
    p.coord[0] = p1.coord[0] - p2.coord[0]
    p.coord[1] = p1.coord[1] - p2.coord[1]
    p.coord[2] = p1.coord[2] - p2.coord[2]
    return p

cdef double norm(Point p):
    return sqrt(p.coord[0]*p.coord[0] + p.coord[1]*p.coord[1] + p.coord[2]*p.coord[2])

cdef double euclidean(Point p1, Point p2):
    return norm(subtract(p1,p2))

cdef double dot(Point v1, Point v2):
    return v1.coord[0]*v2.coord[0] + v1.coord[1]*v2.coord[1] + v1.coord[2]*v2.coord[2]

cdef double get_angle(Point v1, Point v2):
    """
    Returns angle with convention [0,pi]
    """
    norm1 = norm(v1)
    norm2 = norm(v2)
    arg = dot(v1,v2)/(norm1*norm2)
    # This is added to correct for numerical errors
    if arg < -1:
        arg = -1.
    elif arg > 1:
        arg = 1.
    return acos(arg)

@cython.cdivision(True)
cdef double f_cutoff(double r, double gamma, double Rcut):
    """
    Polinomial cutoff function in the, with the steepness determined by "gamma"
    gamma = 2 resembels the cosine cutoff function.
    For large gamma, the function goes towards a step function at Rc.
    """
    if not gamma == 0:
        return 1 + gamma*pow(r/Rcut, gamma+1) - (gamma+1)*pow(r/Rcut, gamma)
    else:
        return 1

@cython.cdivision(True)
cdef double f_cutoff_grad(double r, double gamma, double Rcut):
    if not gamma == 0:
        return gamma*(gamma+1)/Rcut * (pow(r/Rcut, gamma) - pow(r/Rcut, gamma-1))
    else:
        return 0

cdef class Angular_Fingerprint:
    """ comparator for construction of angular fingerprints
    """

    cdef Pool mem
    cdef double Rc1
    cdef double Rc2
    cdef double binwidth1
    cdef double binwidth2
    cdef int Nbins1
    cdef int Nbins2
    cdef double sigma1
    cdef double sigma2
    cdef int nsigma

    cdef double eta
    cdef double gamma
    cdef use_angular

    cdef double volume
    cdef  int Natoms
    cdef int dim

    cdef double m1
    cdef double m2
    cdef double smearing_norm1
    cdef double smearing_norm2

    cdef int Nelements_2body
    cdef int Nelements_3body
    cdef int Nelements
    def __init__(self, atoms, Rc1=4.0, Rc2=4.0, binwidth1=0.1, Nbins2=30, sigma1=0.2, sigma2=0.10, nsigma=4, eta=1, gamma=3, use_angular=True):
        """ Set a common cut of radius
        """
        self.mem = Pool()

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

        self.volume = atoms.get_volume()
        self.Natoms = atoms.get_number_of_atoms()
        self.dim = 3

        # parameters for the binning:
        self.m1 = self.nsigma*self.sigma1/self.binwidth1  # number of neighbour bins included.
        self.smearing_norm1 = erf(1/np.sqrt(2) * self.m1 * self.binwidth1/self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = self.nsigma*self.sigma2/self.binwidth2  # number of neighbour bins included.
        self.smearing_norm2 = erf(1/np.sqrt(2) * self.m2 * self.binwidth2/self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2

        self.Nelements_2body = self.Nbins1
        self.Nelements_3body = self.Nbins2

        if use_angular:
            self.Nelements = self.Nelements_2body + self.Nelements_3body
        else:
            self.Nelements = self.Nelements_2body

    def get_feature(self, atoms):
        """
        """
        cell = atoms.get_cell()
        cdef int Natoms = self.Natoms

        # Get positions and convert to Point-struct
        cdef list pos_np = atoms.get_positions().tolist()
        cdef Point *pos
        pos = <Point*>self.mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            pos[m].coord[0] = pos_np[m][0]
            pos[m].coord[1] = pos_np[m][1]
            pos[m].coord[2] = pos_np[m][2]

        # RADIAL FEATURE

        # Initialize radial feature
        cdef double *feature1
        feature1 = <double*>self.mem.alloc(self.Nelements_2body, sizeof(double))

        cdef int num_pairs, center_bin, minbin_lim, maxbin_lim, newbin
        cdef double Rij, normalization, binpos, c, erfarg_low, erfarg_up, value
        cdef int i, j, n
        for i in range(Natoms):
            for j in range(Natoms):
                Rij = euclidean(pos[i], pos[j])

                # Stop if distance too long or atoms are the same one.
                if Rij > self.Rc1+self.nsigma*self.sigma1 or Rij < 1e-6:
                    continue

                # Calculate normalization
                num_pairs = Natoms*Natoms
                normalization = 1./self.smearing_norm1
                normalization /= 4*M_PI*Rij*Rij * self.binwidth1 * num_pairs/self.volume

                # Identify what bin 'Rij' belongs to + it's position in this bin
                center_bin = <int> floor(Rij/self.binwidth1)
                binpos = Rij/self.binwidth1 - center_bin

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = <int> -ceil(self.m1 - binpos)
                maxbin_lim = <int> ceil(self.m1 - (1-binpos))

                for n in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + n
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 1./sqrt(2)*self.binwidth1/self.sigma1
                    erfarg_low = max(-self.m1, n-binpos)
                    erfarg_up = min(self.m1, n+(1-binpos))
                    value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)

                    # Apply normalization
                    value *= normalization

                    feature1[newbin] += value

        # Convert radial feature to numpy array
        feature1_np = np.zeros(self.Nelements_2body)
        for m in range(self.Nelements_2body):
            feature1_np[m] = feature1[m]

        # Return feature if only radial part is desired
        if not self.use_angular:
            return feature1_np

        # ANGULAR FEATURE

        # Initialize angular feature
        cdef double *feature2
        feature2 = <double*>self.mem.alloc(self.Nelements_3body, sizeof(double))

        cdef Point RijVec, RikVec
        cdef double angle
        cdef int k, cond_ij, cond_ik
        for i in range(Natoms):
            for j in range(Natoms):
                for k in range(j+1, Natoms):
                    Rij = euclidean(pos[i], pos[j])
                    Rik = euclidean(pos[i], pos[k])

                    # Stop if distance too long or atoms are the same one.
                    cond_ij = Rij > self.Rc2 or Rij < 1e-6
                    cond_ik = Rik > self.Rc2 or Rik < 1e-6
                    if cond_ij or cond_ik:
                        continue

                    # Calculate angle
                    RijVec = subtract(pos[j],pos[i])
                    RikVec = subtract(pos[k], pos[i])
                    angle = get_angle(RijVec, RikVec)

                    # Calculate normalization
                    num_pairs = Natoms*Natoms*Natoms
                    normalization = 1./self.smearing_norm2
                    normalization /= num_pairs/self.volume

                    # Identify what bin 'Rij' belongs to + it's position in this bin
                    center_bin = <int> floor(angle/self.binwidth1)
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current atomic distance deltaR.
                    minbin_lim = <int> -ceil(self.m2 - binpos)
                    maxbin_lim = <int> ceil(self.m2 - (1-binpos))

                    for n in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + n

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2-1:
                            newbin = 2*self.Nbins2 - newbin - 1

                        # Calculate gauss contribution to current bin
                        c = 1./sqrt(2)*self.binwidth2/self.sigma2
                        erfarg_low = max(-self.m2, n-binpos)
                        erfarg_up = min(self.m2, n+(1-binpos))
                        value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)
                        value *= f_cutoff(Rij, self.gamma, self.Rc2) * f_cutoff(Rik, self.gamma, self.Rc2)
                        # Apply normalization
                        value *= normalization

                        feature2[newbin] += value

        # Convert angular feature to numpy array
        feature2_np = np.zeros(self.Nelements_3body)
        for m in range(self.Nelements_3body):
            feature2_np[m] = self.eta * feature2[m]

        feature_np = np.zeros(self.Nelements)
        feature_np[:self.Nelements_2body] = feature1_np
        feature_np[self.Nelements_2body:] = feature2_np
        return feature_np

    def get_featureMat(self, atoms_list):
        featureMat = np.array([self.get_feature(atoms) for atoms in atoms_list])
        featureMat = np.array(featureMat)
        return featureMat

    def get_featureGradient(self, atoms):
        cell = atoms.get_cell()
        cdef int Natoms = self.Natoms

        # Get positions and convert to Point-struct
        cdef list pos_np = atoms.get_positions().tolist()
        cdef Point *pos
        pos = <Point*>self.mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            pos[m].coord[0] = pos_np[m][0]
            pos[m].coord[1] = pos_np[m][1]
            pos[m].coord[2] = pos_np[m][2]

        # RADIAL FEATURE GRADIENT

        # Initialize radial feature-gradient
        cdef double *feature_grad1
        feature_grad1 = <double*>self.mem.alloc(self.Nelements_2body * Natoms * self.dim, sizeof(double))

        cdef Point RijVec
        cdef int num_pairs, center_bin, minbin_lim, maxbin_lim, newbin
        cdef double Rij, normalization, binpos, c, arg_low, arg_up, value1, value2, value
        cdef int i, j, n
        for i in range(Natoms):
            for j in range(Natoms):
                Rij = euclidean(pos[i], pos[j])

                # Stop if distance too long or atoms are the same one.
                if Rij > self.Rc1+self.nsigma*self.sigma1 or Rij < 1e-6:
                    continue
                RijVec = subtract(pos[j],pos[i])

                # Calculate normalization
                num_pairs = Natoms*Natoms
                normalization = 1./self.smearing_norm1
                normalization /= 4*M_PI*Rij*Rij * self.binwidth1 * num_pairs/self.volume

                # Identify what bin 'Rij' belongs to + it's position in this bin
                center_bin = <int> floor(Rij/self.binwidth1)
                binpos = Rij/self.binwidth1 - center_bin

                # Lower and upper range of bins affected by the current atomic distance deltaR.
                minbin_lim = <int> -ceil(self.m1 - binpos)
                maxbin_lim = <int> ceil(self.m1 - (1-binpos))

                for n in range(minbin_lim, maxbin_lim + 1):
                    newbin = center_bin + n
                    if newbin < 0 or newbin >= self.Nbins1:
                        continue

                    # Calculate gauss contribution to current bin
                    c = 1./sqrt(2)*self.binwidth1/self.sigma1
                    arg_low = max(-self.m1, n-binpos)
                    arg_up = min(self.m1, n+(1-binpos))
                    value1 = -1./Rij * (erf(c*arg_up) - erf(c*arg_low))
                    value2 = -1./(self.sigma1*sqrt(2*M_PI)) * (exp(-pow(c*arg_up,2)) - exp(-pow(c*arg_low,2)))
                    value = value1 + value2

                    # Apply normalization
                    value *= normalization

                    # Add to the the gradient matrix
                    for m in range(3):
                        feature_grad1[newbin * Natoms*self.dim + self.dim*i+m] += -value/Rij * RijVec.coord[m]
                        feature_grad1[newbin * Natoms*self.dim + self.dim*j+m] += value/Rij * RijVec.coord[m]


        # Convert radial feature to numpy array
        cdef int grad_index
        feature_grad1_np = np.zeros((Natoms*self.dim, self.Nelements_2body))
        for m in range(self.Nelements_2body):
            for grad_index in range(Natoms*self.dim):
                feature_grad1_np[grad_index][m] = feature_grad1[m * Natoms*self.dim + grad_index]

        # Return feature if only radial part is desired
        if not self.use_angular:
            return feature_grad1_np

        # ANGULAR FEATURE-GRADIENT

        # Initialize angular feature-gradient
        cdef double *feature_grad2
        feature_grad2 = <double*>self.mem.alloc(self.Nelements_3body * Natoms * self.dim, sizeof(double))

        cdef Point RikVec, angle_grad_i, angle_grad_j, angle_grad_k
        cdef double angle, cos_angle, a
        cdef int k, cond_ij, cond_ik, bin_index
        for i in range(Natoms):
            for j in range(Natoms):
                for k in range(j+1, Natoms):
                    Rij = euclidean(pos[i], pos[j])
                    Rik = euclidean(pos[i], pos[k])

                    # Stop if distance too long or atoms are the same one.
                    cond_ij = Rij > self.Rc2 or Rij < 1e-6
                    cond_ik = Rik > self.Rc2 or Rik < 1e-6
                    if cond_ij or cond_ik:
                        continue

                    # Calculate angle
                    RijVec = subtract(pos[j],pos[i])
                    RikVec = subtract(pos[k], pos[i])
                    angle = get_angle(RijVec, RikVec)
                    cos_angle = cos(angle)

                    for m in range(3):
                        if not (angle == 0 or angle == M_PI):
                            a = -1/sqrt(1 - cos_angle*cos_angle)
                            angle_grad_j.coord[m] = a * (RikVec.coord[m]/(Rij*Rik) - cos_angle*RijVec.coord[m]/(Rij*Rij))
                            angle_grad_k.coord[m] = a * (RijVec.coord[m]/(Rij*Rik) - cos_angle*RikVec.coord[m]/(Rik*Rik))
                            angle_grad_i.coord[m] = -(angle_grad_j.coord[m] + angle_grad_k.coord[m])
                        else:
                            angle_grad_j.coord[m] = 0
                            angle_grad_k.coord[m] = 0
                            angle_grad_i.coord[m] = 0

                    fc_ij = f_cutoff(Rij, self.gamma, self.Rc2)
                    fc_ik = f_cutoff(Rik, self.gamma, self.Rc2)
                    fc_grad_ij = f_cutoff_grad(Rij, self.gamma, self.Rc2)
                    fc_grad_ik = f_cutoff_grad(Rik, self.gamma, self.Rc2)

                    # Calculate normalization
                    num_pairs = Natoms*Natoms*Natoms
                    normalization = 1./self.smearing_norm2
                    normalization /= num_pairs/self.volume

                    # Identify what bin 'Rij' belongs to + it's position in this bin
                    center_bin = <int> floor(angle/self.binwidth1)
                    binpos = angle/self.binwidth2 - center_bin

                    # Lower and upper range of bins affected by the current atomic distance deltaR.
                    minbin_lim = <int> -ceil(self.m2 - binpos)
                    maxbin_lim = <int> ceil(self.m2 - (1-binpos))

                    for n in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + n

                        # Wrap current bin into correct bin-range
                        if newbin < 0:
                            newbin = abs(newbin)
                        if newbin > self.Nbins2-1:
                            newbin = 2*self.Nbins2 - newbin - 1

                        # Calculate gauss contribution to current bin
                        c = 1./sqrt(2)*self.binwidth2/self.sigma2
                        arg_low = max(-self.m2, n-binpos)
                        arg_up = min(self.m2, n+(1-binpos))
                        value1 = 0.5*erf(c*arg_up)-0.5*erf(c*arg_low)
                        value2 = -1./(self.sigma2*sqrt(2*M_PI)) * (exp(-pow(c*arg_up, 2)) - exp(-pow(c*arg_low, 2)))

                        # Apply normalization
                        value1 *= normalization
                        value2 *= normalization

                        bin_index = newbin * Natoms*self.dim
                        for m in range(3):
                            feature_grad2[bin_index + self.dim*i+m] += -value1 * fc_ik*fc_grad_ij * RijVec.coord[m]/Rij
                            feature_grad2[bin_index + self.dim*j+m] += value1 * fc_ik*fc_grad_ij * RijVec.coord[m]/Rij

                            feature_grad2[bin_index + self.dim*i+m] += -value1 * fc_ij*fc_grad_ik * RikVec.coord[m]/Rik
                            feature_grad2[bin_index + self.dim*k+m] += value1 * fc_ij*fc_grad_ik * RikVec.coord[m]/Rik

                            feature_grad2[bin_index + self.dim*i+m] += value2 * fc_ij * fc_ik * angle_grad_i.coord[m]
                            feature_grad2[bin_index + self.dim*j+m] += value2 * fc_ij * fc_ik * angle_grad_j.coord[m]
                            feature_grad2[bin_index + self.dim*k+m] += value2 * fc_ij * fc_ik * angle_grad_k.coord[m]


        feature_grad2_np = np.zeros((Natoms*self.dim, self.Nelements_3body))
        for m in range(self.Nelements_3body):
            for grad_index in range(Natoms*self.dim):
                feature_grad2_np[grad_index][m] = self.eta * feature_grad2[m * Natoms*self.dim + grad_index]

        feature_grad_np = np.zeros((Natoms*self.dim, self.Nelements))
        feature_grad_np[:, :self.Nelements_2body] = feature_grad1_np
        feature_grad_np[:, self.Nelements_2body:] = feature_grad2_np
        return feature_grad_np

    def get_all_featureGradients(self, atoms_list):
        feature_grads = np.array([self.get_featureGradient(atoms) for atoms in atoms_list])
        feature_grads = np.array(feature_grads)
        return feature_grads
