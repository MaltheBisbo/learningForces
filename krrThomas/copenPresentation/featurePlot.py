import numpy as np
import matplotlib.pyplot as plt


def gauss(r, r0, l):
    return np.exp(-(r-r0)**2/(2*l**2))


pos_x = np.array([0, 0.9, 0.2])
pos_y = np.array([0, 0.8, -0.7])

pos = np.array([pos_x, pos_y])
print(pos)
R12 = np.linalg.norm(pos[:,0] - pos[:,1])
R13 = np.linalg.norm(pos[:,0] - pos[:,2])
R23 = np.linalg.norm(pos[:,1] - pos[:,2])

Npoints = 100
r = np.linspace(0, 2.5, Npoints)

l = 0.13
f1 = gauss(r, R12, l)
f2 = gauss(r, R13, l)
f3 = gauss(r, R23, l)
f4 = f1+f2+f3
f5 = f4/r**2

fs = 19

plt.figure()
plt.subplots_adjust(left=0.15, right=0.95,
                    bottom=0.15, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(np.arange(0, 2, step=0.5), fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('F', fontsize=fs)
plt.xlim([0,2.5])
plt.ylim([0,1.05])

plt.fill_between(r, np.zeros(Npoints), f1, facecolor='blue', alpha=0.2)
plt.savefig('figures/featurePlot1.pdf')


plt.figure()
plt.subplots_adjust(left=0.15, right=0.95,
                    bottom=0.15, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(np.arange(0, 2, step=0.5), fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('F', fontsize=fs)
plt.xlim([0,2.5])
plt.ylim([0,1.05])

plt.fill_between(r, np.zeros(Npoints), f1, facecolor='blue', alpha=0.2)
plt.fill_between(r, np.zeros(Npoints), f2, facecolor='blue', alpha=0.2)
plt.savefig('figures/featurePlot2.pdf')


plt.figure()
plt.subplots_adjust(left=0.15, right=0.95,
                    bottom=0.15, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(np.arange(0, 2, step=0.5), fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('F', fontsize=fs)
plt.xlim([0,2.5])
plt.ylim([0,1.05])

plt.fill_between(r, np.zeros(Npoints), f1, facecolor='blue', alpha=0.2)
plt.fill_between(r, np.zeros(Npoints), f2, facecolor='blue', alpha=0.2)
plt.fill_between(r, np.zeros(Npoints), f3, facecolor='blue', alpha=0.2)
plt.savefig('figures/featurePlot3.pdf')


plt.figure()
plt.subplots_adjust(left=0.15, right=0.95,
                    bottom=0.15, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(np.arange(0, 2, step=0.5), fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('F', fontsize=fs)
plt.xlim([0,2.5])
plt.ylim([0,1.05])

plt.fill_between(r, np.zeros(Npoints), f1, facecolor='blue', alpha=0.2)
plt.fill_between(r, np.zeros(Npoints), f2, facecolor='blue', alpha=0.2)
plt.fill_between(r, np.zeros(Npoints), f3, facecolor='blue', alpha=0.2)
plt.plot(r, f4, lw=2)
plt.savefig('figures/featurePlot4.pdf')



plt.figure()
plt.subplots_adjust(left=0.15, right=0.95,
                    bottom=0.15, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(np.arange(0, 3, step=1), fontsize=fs)
plt.xlabel('r', fontsize=fs)
plt.ylabel('F', fontsize=fs)
plt.xlim([0,2.5])
plt.ylim([0,2.1])

plt.plot(r, f5, lw=2)
plt.savefig('figures/featurePlot5.pdf')






plt.show()
