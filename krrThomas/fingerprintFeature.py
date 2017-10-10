import numpy as np
from scipy.spatial.distance import euclidean
from scipy.special import erf

class fingerprintFeature():
    def __init__(self, X=None, rcut=6, binwidth=0.2, sigma=0.2, nsigma=4):
	"""
        --input--
        X:                                                                                                                                     Contains data. Each row tepresents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
        
        Rcut:
        Cutoff radius
        
        deltaR:
        radial binsize

        sigma:
        standard deviation for gaussian smearing

        nsigma:
        The distance (as the number of standard deviations sigma) at                                                            
        which the gaussian smearing is cut off (i.e. no smearing beyond                                                         
        that distance). 
        """
	self.X = X
        self.rcut = rcut
        self.binwidth = binwidth
        self.sigma = sigma
        self.nsigma = nsigma

        # parameters for the binning:
        m = int(np.ceil(self.nsigma*self.sigma/self.binwidth))
        self.smearing_norm = erf(0.25*np.sqrt(2)*self.binwidth*(2*m+1)*1./self.sigma)
        self.Nbins = int(np.floor(rcut/binwidth))

    def get_singleFeature(self, x):
	"""
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """
        Natoms = int(x.shape[0])
        gamma = Natoms*(Natoms-1)/2
            
        R = self.radiusMatrix(x)
        # filter distances longer than rcut + nsigma*sigma
        R = R[R <= self.rcut + self.nsigma*self.sigma]
        rdf = np.zeros(self.Nbins)
        for i in range(self.Nbins):
             r = (i+1/2)*self.rbin
             
                    

    def radiusMatrix(self, x):
        """
        Calculates the matrix consisting of all pairwise euclidean distances.
        """
        Ndim = 2
        Natoms = int(x.shape[0])
        x = x.reshape((Natoms, Ndim))
        R = np.array([[euclidean(xi, xj) for xj in x] for xi in x])
        return R
    
    def radiusVector(self, x):
        """
        Calculates the matrix consisting of all pairwise euclidean distances.
        """
        Ndim = 2
        Natoms = int(x.shape[0])
	x = x.reshape((Natoms, Ndim))
        Rvec = np.zeros(Natoms*(Natoms-1)/2)
        k = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                Rvec[k] = euclidean(x[i],x[j])
                k += 1
        return Rvec
