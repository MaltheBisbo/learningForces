import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator

class krr():
    def __init__(self, reg, comparator):
        self.reg = reg
        self.comparator = comparator

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
    

def f_target(x):
    return np.cos(x)


kwargs = {'sigma': 1}
comp = gaussComparator(**kwargs)

# set up model
model = krr(reg=1e-4, comparator=comp)




# training data
x_train = np.array([1,3,4,7]).reshape((-1,1))
y_train = f_target(x_train)

# train model
model.train(x_train, y_train, beta=0)

X = np.linspace(0,2.6*np.pi,1000)
y_target = f_target(X)

y_predict1 = []
for x in X:
    x = np.array([x])
    y = model.predict(x)
    y_predict1.append(y)
y_predict1 = np.array(y_predict1)

# train model
model.train(x_train, y_train, beta=3)

y_predict2 = []
for x in X:
    x = np.array([x])
    y = model.predict(x)
    y_predict2.append(y)
y_predict2 = np.array(y_predict2)

# train model
model.train(x_train, y_train, beta=7)

y_predict3 = []
for x in X:
    x = np.array([x])
    y = model.predict(x)
    y_predict3.append(y)
y_predict3 = np.array(y_predict3)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X, y_target, color='k', label='target')
plt.plot(X, y_predict1, label='bias=0')
plt.plot(X, y_predict2, label='bias=3')
plt.plot(X, y_predict3, label='bias=7')
plt.plot(x_train, y_train, 'o')
plt.legend()
plt.show()
