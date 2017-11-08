import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ_energy, doubleLJ_gradient
import matplotlib.pyplot as plt

def Efun(X):
    params = (1.5, 1, np.sqrt(0.02))
    return doubleLJ_energy(X, params[0], params[1], params[2])

def gradfun(X):
    params = (1.5, 1, np.sqrt(0.02))
    return doubleLJ_gradient(X, params[0], params[1], params[2])

def callback(x_cur, obj):
    global Xtraj
    Xtraj.append(x_cur)
    print(obj.fk)

def localMinimizer(X):
    global Xtraj
    Xtraj = []
    res = minimize(Efun, X, method="BFGS",jac=gradfun, tol=1e-5, callback=callback)
    Xtraj = np.array(Xtraj)
    return res, Xtraj


x = np.array([0, 0,
              0, 1.2,
              0.9, 0.2,
              1.3, 1.1])

res, Xtraj = localMinimizer(x)
xrel = Xtraj[-1]
Etraj = np.array([Efun(x) for x in Xtraj])

plt.figure(1)
plt.scatter(x[0::2], x[1::2], color='r')
plt.scatter(xrel[0::2], xrel[1::2], color='b')
plt.gca().set_aspect('equal', adjustable='box')

plt.figure(2)
plt.plot(np.arange(len(Etraj)), Etraj)
plt.show()
