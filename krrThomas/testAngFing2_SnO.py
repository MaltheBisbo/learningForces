import numpy as np
from angular_fingerprintFeature2 import Angular_Fingerprint
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

atoms = read('fromThomas/data_SnO.traj', index=':')

a = atoms[0]
featureCalculator = Angular_Fingerprint(a, Rc1=6)

res1, res2 = featureCalculator.get_features(a)

print(res1)

view(a)

plt.figure(1)
plt.plot(np.arange(len(res1[(8,8)])), res1[(8,8)])
plt.plot(np.arange(len(res1[(8,50)])), res1[(8, 50)])
plt.plot(np.arange(len(res1[(50,50)])), res1[(50,50)])
plt.show()
