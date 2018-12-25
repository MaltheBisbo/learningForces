import numpy as np
from scipy.spatial.distance import sqeuclidean
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


class maternComparator():
    def __init__(self, featureCalculator=None, max_looks_like_dist=0.5, **kwargs):
        self.featureCalculator = featureCalculator
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        self.max_looks_like_dist = max_looks_like_dist

    def set_args(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']

    def get_similarity_matrix(self, featureMat=None):
        if featureMat is not None:
            self.featureMat = featureMat
        else:
            print("You need to supply a feature matrix")
        
        d = cdist(self.featureMat, self.featureMat, metric='euclidean')
        prefac = 1 + np.sqrt(5)*d/self.sigma + 5*d**2/(3*self.sigma**2)
        self.similarityMat = prefac * np.exp(-np.sqrt(5)*d/self.sigma)
        return self.similarityMat

    def get_similarity_vector(self, fnew, featureMat=None, split=False):
        if featureMat is not None:
            self.featureMat = featureMat
        d = cdist(fnew.reshape((1,len(fnew))), self.featureMat, metric='euclidean')
        prefac = 1 + np.sqrt(5)*d/self.sigma + 5*d**2/(3*self.sigma**2)
        self.similarityVec = (prefac * np.exp(-np.sqrt(5)*d/self.sigma)).reshape(-1)

        if split:
            return self.similarityVec, d
        else:
            return self.similarityVec

    def single_comparison(self, feature1, feature2, sigma=None):
        if sigma is None:
            sigma = self.sigma
        d = sqeuclidean(feature1, feature2)
        prefac = 1 + np.sqrt(5)*d/self.sigma + 5*d**2/(3*self.sigma**2)
        return prefac * np.exp(-np.sqrt(5)*d/self.sigma)

    def get_jac(self, fnew, kappa=None, featureMat=None):
        """
        Calculates tor jacobian of the similarity vector 'k' with respect
        to the feature vector 'f' of the new data-point.
        ie. calculates dk_df.
        Using the chain rule: dk_df = dk_dd*dd_df , where d is the distance measure
        """
        if featureMat is None:
            featureMat = self.featureMat.copy()

        d = cdist(fnew.reshape((1,len(fnew))), self.featureMat, metric='euclidean').reshape(-1)

        deriv_part1 = (5/(3*self.sigma**2) + 5*np.sqrt(5)*d/(3*self.sigma**3)) * np.exp(-np.sqrt(5)*d/self.sigma)
        feature_difference = (featureMat - fnew.reshape((1, fnew.shape[0])))

        dk_df = deriv_part1.reshape((-1,1)) * feature_difference
        
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
