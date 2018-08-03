import numpy as np
import matplotlib.pyplot as plt


x = [1,2,3,4]
frac = np.array([0.4, 0.4, 0.6, 0.8])

plt.figure(figsize=(7,4))
plt.subplots_adjust(bottom=0.15)
plt.xlabel('Structure', fontsize=19)
plt.ylabel('Discovery rate', fontsize=19)
plt.ylim([0,1])
plt.xticks(np.arange(0, 5, step=1), fontsize=16)
plt.yticks(np.arange(0, 1.1, step=0.5), fontsize=16)
plt.bar(x, frac, width=0.3, color='steelblue')
plt.savefig('fullerene_succes.pdf')
plt.show()
