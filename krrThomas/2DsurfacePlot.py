import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(binwidth1_array, sigma1_array)

surf = ax.plot_surface(X, Y, FVU_grid.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.text2D(0.05, 0.96, 'graphene prediction MAE for Ntrain=137 \n using the radial fingerprint with Rc1=5', trans\
form=ax.transAxes, size=14)
ax.set_xlabel('binwidth1')
ax.set_ylabel('sigma1')
ax.set_zlabel('MAE')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
