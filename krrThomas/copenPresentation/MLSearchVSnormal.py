import numpy as np
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator


def f_target(x):
    x += 0.8
    width = 1.2
    x0 = 5.8
    d = np.abs(x-x0)
    return np.cos(1.2*x) - 0.3*x*np.exp(-d**2/(2*width**2)) - 2 + 0.7*np.sin(4*x)

X = np.linspace(0,10,1000)
y_target = f_target(X)

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.0
fs = 16

plt.figure(figsize=(7,4))
plt.xticks(np.arange(0, 12, step=2), fontsize=fs)
plt.yticks(np.arange(-4, 1, step=1), fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('Energy', fontsize=fs)
plt.xlim([0,9.6])
plt.ylim([-4.5,-1.3])
plt.plot(X, y_target, color='darkgreen', lw=2.5)
plt.axis('off')
plt.savefig('energyLandscape.pdf', transparent=True)
plt.show()
