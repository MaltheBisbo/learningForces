import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.special import binom
from scipy.misc import factorial as fac

class maternComparator():
    def __init__(self, featureMat=None, **kwargs):
        self.featureMat = featureMat
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        self.n = 2

    def set_args(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']

    def get_kernelMat(self, featureMat=None):
        if featureMat is not None:
            self.featureMat = featureMat
        v = self.n + 0.5
        c = np.sqrt(2*v)/self.sigma
            
        d = cdist(self.featureMat, self.featureMat, metric='euclidean')
        kernelMat = np.exp(-c*d) * self.get_Pn(d)
        
        return kernelMat
    
    def get_kernelVec(self, f, featureMat=None, split=False):
        if featureMat is not None:
            self.featureMat = featureMat
        v = self.n + 0.5
        
        d = cdist(f.reshape((1,len(f))), self.featureMat, metric='euclidean')
        k1 = np.exp(-np.sqrt(2*v)/self.sigma * d)
        k2 = self.get_Pn(d)

        if not split:
            return k1 * k2
        else:
            return k1, k2, d

    def get_kernel(self, f1, f2, sigma=None, split=False):
        if sigma is None:
            sigma = self.sigma
        v = self.n + 0.5
        c = np.sqrt(2*v)/self.sigma
        
        d = euclidean(f1, f2)
        k1 = np.exp(-c*d)
        k2 = self.get_Pn(d)

        if not split:
            return k1 * k2
        else:
            return k1, k2, d

    def get_kernelVec_Jac(self, f, featureMat):
        """
        Calculates the jacobian of the similarity vector 'k' with respect
        to the feature vector 'f' of the new data-point.
        ie. calculates dk_df.
        Using the chain rule: dk_df = dk_dd*dd_df , where d is the distance measure
        """
        k1Vec, k2Vec, dVec = self.get_kernelVec(f, featureMat, split=True)
        v = self.n + 0.5
        
        c = np.sqrt(2*v)/self.sigma
        dk1Vec_dd = -c*k1Vec
        dk2Vec_dd = self.get_Pn_deriv(dVec)
        
        dkVec_dd = (k1Vec*dk2Vec_dd + k2Vec*dk1Vec_dd).reshape((len(k1Vec), 1))
        dd_df = -(featureMat - f.reshape((1, len(f)))) / dVec.reshape((1,-1))

        dk_df = np.multiply(dkVec_dd, dd_df)
        return dk_df

    def get_kernel_Jac(self, f1, f2):
        """
        Calculates the jacobian of the similarity vector 'k' with respect
        to the feature vector 'f' of the new data-point.
        ie. calculates dk_df.
        Using the chain rule: dk_df = dk_dd*dd_df , where d is the distance measure
        """
        k1, k2, d = self.get_kernel(f1, f2, split=True)
        v = self.n + 0.5
        c = np.sqrt(2*v)/self.sigma
        
        dk1_dd = -c*k1
        dk2_dd = self.get_Pn_deriv(d)
        
        dk_dd = (k1*dk2_dd + k2*dk1_dd)
        dd_df = -(f2 - f1)/d

        dk_df = np.multiply(dk_dd, dd_df)
        return dk_df
    
    def get_kernel_Hess(self, f1, f2):
        """
        Calculated the hessian of the kernel function with respect to
        the two features f1 and f2.
        ie. calculates: d^2/df1df2(kernel)
        """
        Nf = len(f1)
        v = self.n + 0.5
        c = np.sqrt(2*v)/self.sigma
        
        # kernel between the two features
        k1, k2, d = self.get_kernel(f1, f2, split=True)

        dk1_dd = -c*k1
        dk2_dd = self.get_Pn_deriv(d)

        d2k1_d2d = c**2*k1
        d2k2_d2d = self.get_Pn_2deriv(d)
        
        dd_df1 = -1/d*(f2-f1).reshape((Nf,1))
        dd_df2 = -dd_df1
        d2d_df1df2 = np.outer(f2-f1, f2-f1)/d**3 - 1/d*np.identity(Nf)

        grad_f1f2 = np.outer(dd_df1, dd_df2.T)
        Hess1 = k2 * (d2k1_d2d*grad_f1f2 + dk1_dd*d2d_df1df2)
        Hess2 = k1 * (d2k2_d2d*grad_f1f2 + dk2_dd*d2d_df1df2)
        Hess3 = 2 * dk1_dd * dk2_dd * grad_f1f2
        Hess = Hess1 + Hess2 + Hess3
        
        return Hess

    def get_Pn(self, d):
        n = self.n
        v = n + 0.5
        arg = 2*np.sqrt(2*v)/self.sigma * d
        
        Pn = 0
        for k in range(n+1):
            Pn += fac(n+k)/fac(2*n) * binom(n,k) * arg**(n-k)
        return Pn
            
    def get_Pn_deriv(self, d):
        n = self.n
        v = n + 0.5
        c = np.sqrt(2*v)/self.sigma
        arg = 2*c * d
        
        dPn_dd = 0
        for k in range(n):
            dPn_dd += fac(n+k)/fac(2*n) * binom(n,k) * (n-k) * arg**(n-k-1)
        dPn_dd *= 2*c
        return dPn_dd

    def get_Pn_2deriv(self, d):
        n = self.n
        v = n + 0.5
        c = np.sqrt(2*v)/self.sigma
        arg = 2*c * d
        
        d2Pn_d2d = 0
        for k in range(n-1):
            d2Pn_d2d += fac(n+k)/fac(2*n) * binom(n,k) * (n-k) * (n-k-1) * arg**(n-k-2)
        d2Pn_d2d *= 4*c**2
        return d2Pn_d2d
            
