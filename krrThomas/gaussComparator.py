import numpy as np
from scipy.spatial.distance import sqeuclidean
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


class gaussComparator():
#    def __init__(self, featureMat=None, amplitude=1, **kwargs):
#        self.featureMat = featureMat
#        self.amplitude = amplitude
    def __init__(self, featureCalculator=None, amplitude=1, max_looks_like_dist=0.5, **kwargs):
        self.featureCalculator = featureCalculator
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        self.amplitude = amplitude
        self.max_looks_like_dist = max_looks_like_dist

    def set_args(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'amplitude' in kwargs:
            self.amplitude = kwargs['amplitude']

    def hyper_grad(self, featureMat):
        d = cdist(self.featureMat, self.featureMat, metric='sqeuclidean')
        self.similarityMat = self.amplitude*np.exp(-1/(2*self.sigma**2)*d)
        
        hyper_grad_amp = self.similarityMat / self.amplitude
        hyper_grad_sigma = d/self.sigma**3*self.similarityMat
        return np.array([hyper_grad_amp, hyper_grad_sigma])

    def get_similarity_matrix(self, featureMat=None, d=None):
        if featureMat is not None:
            self.featureMat = featureMat
        else:
            print("You need to supply a feature matrix")

        if d is None:
            d = cdist(self.featureMat, self.featureMat, metric='sqeuclidean')
        self.similarityMat = self.amplitude*np.exp(-1/(2*self.sigma**2)*d)
        return self.similarityMat

    def get_similarity_vector(self, fnew, featureMat=None, d=None):
        if featureMat is not None:
            self.featureMat = featureMat
        if d is None:
            d = cdist(fnew.reshape((1,len(fnew))), self.featureMat, metric='sqeuclidean')
        self.similarityVec = self.amplitude*np.exp(-1/(2*self.sigma**2)*d).reshape(-1)

        return self.similarityVec

    def single_comparison(self, feature1, feature2, sigma=None, d=None):
        if sigma is None:
            sigma = self.sigma
        if d is None:
            d = sqeuclidean(feature1, feature2)
        return self.amplitude*np.exp(-1/(2*sigma**2)*d)

    def get_jac(self, fnew, featureMat=None, d=None):
        """
        Calculates tor jacobian of the similarity vector 'k' with respect
        to the feature vector 'f' of the new data-point.
        ie. calculates dk_df.
        Using the chain rule: dk_df = dk_dd*dd_df , where d is the distance measure
        """
        if featureMat is None:
            featureMat = self.featureMat.copy()

        if d is not None:
            kappa = self.get_similarity_vector(fnew, featureMat, d=d)
        else:
            kappa = self.get_similarity_vector(fnew, featureMat)
            
        dk_dd = -1/(2*self.sigma**2)*kappa.reshape((kappa.shape[0], 1))
        dd_df = -2*(featureMat - fnew.reshape((1, fnew.shape[0])))

        dk_df = np.multiply(dk_dd, dd_df)
        return dk_df

    def get_jac_new(self, fnew, featureMat):
        kappa = self.get_similarity_vector(fnew, featureMat)
        dk_dd = -1/(2*self.sigma**2)*kappa.reshape((kappa.shape[0], 1))
        dd_df = -2*(featureMat - fnew.reshape((1, fnew.shape[0])))

        dk_df = np.multiply(dk_dd, dd_df)
        return dk_df
    
    def get_single_Hess(self, f1, f2):
        """
        Calculated the hessian of the kernel function with respect to
        the two features f1 and f2.
        ie. calculates: d^2/df1df2(kernel)
        """
        # kernel between the two features
        kernel = self.single_comparison(f1, f2)

        Nf = f1.shape[0]

        dd_df1 = -2*(f2-f1).reshape((Nf,1))
        dd_df2 = -dd_df1
        d2d_df1df2 = -2*np.identity(Nf)
        u = 1/(2*self.sigma**2)

        Hess = -u*kernel * (u*np.outer(dd_df1, dd_df2.T) - d2d_df1df2)
        return Hess

    def looks_like(self, a1, a2, featureCalculator=None):
        if featureCalculator is None:
            featureCalculator = self.featureCalculator
        
        f1 = featureCalculator.get_feature(a1)
        f2 = featureCalculator.get_feature(a2)
        distance = euclidean(f1, f2)
        if distance < self.max_looks_like_dist:  # Hard coded value
            return True
        else:
            return False

    def __add__(self, kernel_b):
        return Sum(self, kernel_b)

    def __radd__(self, kernel_b):
        return Sum(self, kernel_b)

    def __mul__(self, const):
        return Product(self, const)

    def __rmul__(self, const):
        return Product(self, const)


class Sum():
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_similarity_matrix(self, featureMat=None, d=None):
        if d is None:
            d = cdist(featureMat, featureMat, metric='sqeuclidean')
        return self.k1.get_similarity_matrix(featureMat=featureMat, d=d) + self.k2.get_similarity_matrix(featureMat=featureMat, d=d)

    def get_similarity_vector(self, fnew, featureMat=None, d=None):
        if d is None:
            d = cdist(fnew.reshape((1,len(fnew))), featureMat, metric='sqeuclidean')
        return self.k1.get_similarity_vector(fnew, featureMat=featureMat, d=d) + self.k2.get_similarity_vector(fnew, featureMat=featureMat, d=d)

    def single_comparison(self, feature1, feature2, sigma=None, d=None):
        if d is None:
            d = sqeuclidean(feature1, feature2)
        return self.k1.single_comparison(feature1, feature2, d=d) + self.k2.single_comparison(feature1, feature2, d=d)

    def get_jac(self, fnew, featureMat=None, d=None):
        if d is None:
            d = cdist(fnew.reshape((1,len(fnew))), featureMat, metric='sqeuclidean')
        return self.k1.get_jac(fnew, featureMat=featureMat, d=d) + self.k2.get_jac(fnew, featureMat=featureMat, d=d)

    def get_jac_new(self, fnew, featureMat):
        pass

    def get_single_Hess(self, f1, f2):
        pass

    def looks_like(self, a1, a2, featureCalculator=None):
        return self.k1.looks_like(a1,a2,featureCalculator=featureCalculator)

    def __add__(self, kernel_b):
        return Sum(self, kernel_b)

    def __radd__(self, kernel_b):
        return Sum(self, kernel_b)

    def __mul__(self, const):
        return Product(self, const)

    def __rmul__(self, const):
        return Product(self, const)
    

class Product():
    def __init__(self, k, const):
        self.k = k
        self.const = const

    def get_similarity_matrix(self, featureMat=None, d=None):
        return self.const * self.k.get_similarity_matrix(featureMat=featureMat, d=d)

    def get_similarity_vector(self, fnew, featureMat=None, d=None):
        return self.const * self.k.get_similarity_vector(fnew, featureMat=featureMat, d=d)

    def single_comparison(self, feature1, feature2, sigma=None, d=None):
        return self.const * self.k.single_comparison(feature1, feature2)

    def get_jac(self, fnew, featureMat=None, d=None):
        return self.const * self.k.get_jac(fnew, featureMat=featureMat, d=d)

    def get_jac_new(self, fnew, featureMat):
        pass

    def get_single_Hess(self, f1, f2):
        pass

    def looks_like(self, a1, a2, featureCalculator=None):
        return self.k.looks_like(a1,a2,featureCalculator=featureCalculator)

    def __add__(self, kernel_b):
        return Sum(self, kernel_b)

    def __radd__(self, kernel_b):
        return Sum(self, kernel_b)

    def __mul__(self, const):
        return Product(self, const)

    def __rmul__(self, const):
        return Product(self, const)


