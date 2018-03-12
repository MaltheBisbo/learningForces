import numpy as np
import matplotlib.pyplot as plt

from maternComparator import maternComparator

matern = maternComparator()

kwargs = {'sigma': 1}
matern.set_args(**kwargs)
f0 = np.array([0])
f1 = np.array([1])

Npoints = 1000
x_array = np.linspace(0,2,Npoints)
kernel = np.zeros(Npoints)
kernel_Jac = np.zeros((Npoints,2))
kernel_Hess = np.zeros((Npoints,2,2))
x0=np.array([0,0])
for i, x in enumerate(x_array):
    x1=np.array([x,0])
    kernel[i] = matern.get_kernel(x0,x1)
    kernel_Jac[i] = matern.get_kernel_Jac(x0,x1.reshape((1,2)))
    kernel_Hess[i] = matern.get_kernel_Hess(x0,x1)
    
plt.figure()
plt.plot(x_array, kernel)

plt.figure()
plt.plot(x_array, kernel_Jac)

plt.figure()
plt.plot(x_array, kernel_Hess.reshape((Npoints,-1)))
plt.show()
