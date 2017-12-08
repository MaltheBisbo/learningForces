import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

Rc1 = 7
binwidth1_array = np.linspace(0.02, 0.2, 10)
sigma1_array = np.linspace(0.1, 1, 10)
MAEgrid = np.zeros((10, 10))
for i, binwidth1 in enumerate(binwidth1_array):
    for j, sigma1 in enumerate(sigma1_array):
        MAEcurve = np.loadtxt('Rc{0:d}/SnO_radialResults_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}.txt'.format(Rc1, binwidth1, sigma1), delimiter='\t')
        MAEgrid[j,i] = MAEcurve[-11]

fig = plt.figure()
ax = fig.gca(projection='3d')

binwidth1_array, sigma1_array = np.meshgrid(binwidth1_array, sigma1_array)

# Plot the surface.
surf = ax.plot_surface(binwidth1_array, sigma1_array, MAEgrid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

#ax.text2D(0.2, 0.95, 'Radial fingerprint: Rcut=7, no fcut', transform=ax.transAxes)
ax.set_title('Radial fingerprint: Rcut=7, no fcut')
ax.set_xlabel('binwidth')
ax.set_ylabel('sigma')
ax.set_zlabel('MAE')
plt.show()
