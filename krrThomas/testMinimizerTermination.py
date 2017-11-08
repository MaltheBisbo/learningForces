import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ
import matplotlib.pyplot as plt

def callback(x_cur):
    global Xtraj
    Xtraj.append(x_cur)

def localMinimizer(X):
    global Xtraj
    Xtraj = []
    res = minimize(doubleLJ, X, method="BFGS", tol=1e-5, callback=callback)
    Xtraj = np.array(Xtraj)
    return res, Xtraj

x = np.array([0, 0,
              0, 1.2,
              0.9, 0.2,
              1.3, 1.1])

res, xrel = localMinimizer(x)


plt.scatter(x[0::2], x[1::2], color='r')
plt.scatter(xrel[-1][0::2], xrel[-1][1::2], color='b')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
