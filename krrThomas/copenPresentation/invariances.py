import numpy as np
from doubleLJ import doubleLJ_energy_ase as E_doubleLJ
import matplotlib.pyplot as plt

from ase import Atoms


def structure(x1,x2):
    '''
    x1, x2 is the two new coordiantes
    d0 is the equilibrium two body distance
    '''
    d0 = 1.07
    theta1 = -np.pi/3
    theta2 = -2/3*np.pi
    pos0 = np.array([0,0,0])
    pos1 = np.array([-d0 + x1*np.cos(theta1), x1*np.sin(theta1), 0])
    pos2 = np.array([d0 + x2*np.cos(theta2), x2*np.sin(theta2), 0])

    pos = np.array([pos1, pos0, pos2])

    a = Atoms('3He',
              positions=pos,
              pbc=[0,0,0],
              cell=[3,3,3])
    return a


pos_x = np.array([0, 0.9, 0.2])
pos_y = np.array([0, 0.8, -0.7])
pos_x -= np.mean(pos_x)
pos_y -= np.mean(pos_y)

fs = 15

plt.figure(figsize=(2.0,2.0))
plt.subplots_adjust(left=0.3, right=0.99,
                    bottom=0.3, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)

plt.xlim([-1,1.5])
plt.ylim([-1.3,1.3])
plt.plot(pos_x, pos_y, 'ro', ms=7)

plt.text(pos_x[0]-0.4, pos_y[0]-0.1, '{}'.format(1), fontsize=fs)
plt.text(pos_x[1]+0.2, pos_y[1]+0.0, '{}'.format(2), fontsize=fs)
plt.text(pos_x[2]-0.3, pos_y[2]-0.4, '{}'.format(3), fontsize=fs)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figures/invariance_base.pdf')



plt.figure(figsize=(2.7,2.0))
plt.subplots_adjust(left=0.22, right=0.99,
                    bottom=0.3, top=0.99)
plt.xticks(np.arange(-1, 3, step=1), fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)

plt.xlim([-1, 2.5])
plt.ylim([-1.3, 1.3])
plt.plot(pos_x, pos_y, 'ro', alpha=0.2)

pos_x_trans = pos_x + 1.3
plt.plot(pos_x_trans, pos_y, 'ro')

#plt.text(pos_x_trans[0]-0.4, pos_y[0]-0.1, '{}'.format(1), fontsize=fs)
#plt.text(pos_x_trans[1]+0.2, pos_y[1]+0.0, '{}'.format(2), fontsize=fs)
#plt.text(pos_x_trans[2]-0.3, pos_y[2]-0.4, '{}'.format(3), fontsize=fs)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figures/invariance_trans.pdf')



plt.figure(figsize=(2.0,2.0))
plt.subplots_adjust(left=0.3, right=0.99,
                    bottom=0.3, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)

plt.xlim([-1, 1.5])
plt.ylim([-1.3, 1.3])
plt.plot([pos_x[2], pos_x[0], pos_x[1]], [pos_y[2], pos_y[0], pos_y[1]], 'k-', alpha=0.2)
plt.plot(pos_x, pos_y, 'ro', alpha=0.2)

angle = -np.pi/4

pos_x_rot = (pos_x-pos_x[0])*np.cos(angle) - (pos_y-pos_y[0])*np.sin(angle) + pos_x[0]
pos_y_rot = (pos_x-pos_x[0])*np.sin(angle) + (pos_y-pos_y[0])*np.cos(angle) + pos_y[0]
#pos_y_rot = pos_x*np.sin(angle) + pos_y*np.cos(angle)
plt.plot([pos_x_rot[2], pos_x_rot[0], pos_x_rot[1]], [pos_y_rot[2], pos_y_rot[0], pos_y_rot[1]], 'k-')
plt.plot(pos_x_rot, pos_y_rot, 'ro')


plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figures/invariance_rot.pdf')



plt.figure(figsize=(2.0,2.0))
plt.subplots_adjust(left=0.3, right=0.99,
                    bottom=0.3, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)

plt.xlim([-1, 1.5])
plt.ylim([-1.3, 1.3])
plt.plot(pos_x, pos_y, 'ro', alpha=0.2)

x0 = 0.0
pos_x_refl = pos_x + 2*(x0 - pos_x)
plt.plot(pos_x_refl, pos_y, 'ro')
plt.plot([x0,x0], [-1.3, 1.3], 'k--')

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figures/invariance_reflect.pdf')



plt.figure(figsize=(2.0,2.0))
plt.subplots_adjust(left=0.3, right=0.99,
                    bottom=0.3, top=0.99)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)

plt.xlim([-1,1.5])
plt.ylim([-1.3,1.3])
plt.plot(pos_x, pos_y, 'ro')

plt.text(pos_x[0]-0.4, pos_y[0]-0.1, '{}'.format(3), fontsize=fs)
plt.text(pos_x[1]+0.2, pos_y[1]+0.0, '{}'.format(2), fontsize=fs)
plt.text(pos_x[2]-0.3, pos_y[2]-0.4, '{}'.format(1), fontsize=fs)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figures/invariance_permut.pdf')

plt.show()
