import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator

class krr():
    def __init__(self, reg, comparator):
        self.reg = reg
        self.comparator = comparator
        self.sigma = self.comparator.sigma
        
    def train(self,x_train,y_train, beta=0):
        self.x_train = x_train
        kernelMat = comp.get_similarity_matrix(x_train)

        self.beta = beta
        
        A = kernelMat + self.reg*np.identity(len(y_train))
        self.Ainv = np.linalg.inv(A)
        self.alpha = np.dot(self.Ainv, y_train - beta)

    def predict(self, x):
        kernelVec = comp.get_similarity_vector(x, self.x_train)

        y_predict = kernelVec.dot(self.alpha) + self.beta
        return y_predict
    
    def plotModel(self, x_min, x_max):
        #plt.figure()
        for x0, alpha in zip(self.x_train, self.alpha):
            x_plot = np.linspace(x0-3.5*self.sigma, x0+3.5*self.sigma, 100)
            d = np.abs(x_plot - x0)
            y = alpha*np.exp(-d**2/(2*self.sigma**2))
            plt.plot(x_plot,y, 'k:')
        
    
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
    
kwargs = {'sigma': 1}
comp = gaussComparator(**kwargs)

# set up model
model = krr(reg=1e-4, comparator=comp)




# training data
x_train = np.array([1,2,2.5,3]).reshape((-1,1))
#x_train = np.array([1,3,5,7]).reshape((-1,1))
y_train = f_target(x_train)

# train model
model.train(x_train, y_train, beta=0)

X = np.linspace(0,10,1000)
y_target = f_target(X)

y_predict1 = []
for x in X:
    x = np.array([x])
    y = model.predict(x)
    y_predict1.append(y)
y_predict1 = np.array(y_predict1)

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.0
fs =14

plt.figure()
plt.xticks(np.arange(0, 12, step=2), fontsize=fs)
plt.yticks(np.arange(-4, 1, step=1), fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('f', fontsize=fs)
plt.xlim([0,10])
plt.ylim([-4.5,0])
model.plotModel(0,1)
plt.plot(X, y_target, color='k', label='True', lw=2)
plt.plot(X, y_predict1, color='steelblue', label='Model')
plt.plot(x_train, y_train, 'ro')
plt.legend(loc=3, fontsize=fs)
plt.savefig('figures/krrExample5.pdf')
plt.show()
