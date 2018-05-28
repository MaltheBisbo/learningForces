import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator

class krr():
    def __init__(self, reg, comparator):
        self.reg = reg
        self.comparator = comparator
        self.sigma = self.comparator.sigma
        
    def train(self,x_train,y_train, beta=0):
        self.Ndata = len(x_train)
        self.x_train = x_train
        self.y_train = y_train
        
        kernelMat = comp.get_similarity_matrix(x_train)

        self.beta = beta
        
        A = kernelMat + self.reg*np.identity(len(y_train))
        self.Ainv = np.linalg.inv(A)
        self.alpha = np.dot(self.Ainv, y_train - beta)

    def predict(self, x, return_error=False):
        kernelVec = comp.get_similarity_vector(x, self.x_train)

        y_predict = kernelVec.dot(self.alpha) + self.beta

        if return_error:
            alpha_err = np.dot(self.Ainv, kernelVec)
            theta0 = np.dot(self.y_train.T, self.alpha) / self.Ndata
            error = np.sqrt(np.abs(theta0*(1 - np.dot(kernelVec, alpha_err))))
            return y_predict, error
        else:
            return y_predict
    
    def plotModel(self, x_min, x_max):
        #plt.figure()
        for x0, alpha in zip(self.x_train, self.alpha):
            x_plot = np.linspace(x0-3.5*self.sigma, x0+3.5*self.sigma, 100)
            d = np.abs(x_plot - x0)
            y = alpha*np.exp(-d**2/(2*self.sigma**2))
            plt.plot(x_plot,y, 'k:', lw=1.5)
        
    
"""
def f_target(x):
    x0_array = [1,2,7.2,5.3]
    width_array = [0.5, 0.9,0.8,1.2]
    height_array = [0.5,1,0.9,1.2]
    f = 0
    for x0, width, height in zip(x0_array, width_array, height_array):
        d = abs(x - x0)
        f += -height*np.exp(-d**2/(2*width**2))
    return f
"""
def f_target(x):
    width = 1.2
    x0 = 5.8
    d = np.abs(x-x0)
    return np.cos(1.2*x) - 0.3*x*np.exp(-d**2/(2*width**2)) - 2

#def f_target(x):
#    return np.cos(0.3*x) + 1


kwargs = {'sigma': 0.5}
comp = gaussComparator(**kwargs)

# set up model
model = krr(reg=1e-4, comparator=comp)




# training data
#x_train = np.array([1,2,2.5,3]).reshape((-1,1))
x_train = np.array([1,3,5,7]).reshape((-1,1))
y_train = f_target(x_train)

# train model
model.train(x_train, y_train, beta=0)

X = np.linspace(0,10,1000)
y_target = f_target(X)

y_predict = []
error_array = []
for x in X:
    x = np.array([x])
    y, error = model.predict(x, return_error=True)
    y_predict.append(y)
    error_array.append(error)
y_predict = np.array(y_predict).reshape(-1)
error_array = np.array(error_array).reshape(-1)

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.0
fs = 16

plt.figure()
plt.xticks(np.arange(0, 12, step=2), fontsize=fs)
plt.yticks(np.arange(-4, 1, step=1), fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('Energy', fontsize=fs)
plt.xlim([0,10])
plt.ylim([-4.5,0])
model.plotModel(0,1)
plt.plot(X, y_target, color='darkgreen', label='True', lw=2.5)
plt.plot(X, y_predict, color='mediumblue', label='Model', lw=1.5)
plt.plot(x_train, y_train, 'ro')

print(X.shape)
print(y_predict.shape)
print(error_array.shape)
#plt.fill_between(X, y_predict+0.5*error_array, y_predict-0.5*error_array, facecolor='blue', alpha=0.2)

plt.legend(loc=3, fontsize=fs)
plt.savefig('figures/krrExample5.pdf')
plt.show()
