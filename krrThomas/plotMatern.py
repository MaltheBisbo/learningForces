import numpy as np
import matplotlib.pyplot as plt

from maternComparator import maternComparator

matern = maternComparator()

kwargs = {'sigma': 1}
matern.set_args(**kwargs)
f0 = np.array([0])
f1 = np.array([1])

Npoints = 4000
x_array = np.linspace(-2,2,Npoints)

kernel = np.zeros(Npoints)
kernel_Jac = np.zeros((Npoints,2))
kernel_Hess = np.zeros((Npoints,2,2))

Pn = np.zeros(Npoints)
Pn_deriv = np.zeros(Npoints)
Pn_2deriv = np.zeros(Npoints)

x0 = np.array([0,0])
for i, x in enumerate(x_array):
    x1 = np.array([x,x])

    kernel[i] = matern.get_kernel(x1,x0)
    kernel_Jac[i] = matern.get_kernel_Jac(x1,x0)
    kernel_Hess[i] = matern.get_kernel_Hess(x1,x0)

    #d = x
    #Pn[i] = matern.get_Pn(d)
    #Pn_deriv[i] = matern.get_Pn_deriv(d)
    #Pn_2deriv[i] = matern.get_Pn_2deriv(d)

dx = x_array[1] = x_array[0]
plt.figure()
plt.plot(x_array, kernel)

plt.figure()
plt.plot(x_array, kernel_Jac)
plt.plot(x_array, np.gradient(kernel, x_array), linestyle=':')
plt.plot(x_array[:-1] + dx/2, (kernel[1:]-kernel[:-1])/dx)

plt.figure()
plt.plot(x_array, kernel_Hess.reshape((Npoints,-1))[:,3])
plt.plot(x_array, np.gradient(np.gradient(kernel, x_array), x_array), linestyle=':')

"""
plt.figure()
plt.plot(x_array, Pn)

plt.figure()
plt.plot(x_array, Pn_deriv)
plt.plot(x_array, np.gradient(Pn, x_array), linestyle=':')

plt.figure()
plt.plot(x_array, Pn_2deriv)
plt.plot(x_array, np.gradient(np.gradient(Pn, x_array), x_array), linestyle=':')
"""
plt.show()
