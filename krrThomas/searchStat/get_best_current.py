#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

import sys

runs_name = sys.argv[1]
try:
    num = sys.argv[2]
except:
    num = 30

best_structures = []
for i in range(50):
    try:
        traj_path = runs_name + '/run{}/global{}_current.traj'.format(i,i)
        pop_current = read(traj_path, index='-{}:'.format(num))
        E_pop = np.array([a.get_potential_energy() for a in pop_current])
        index_best = np.argmin(E_pop)
        Ebest = E_pop[index_best]
        a_best = pop_current[index_best]

        Ncalc = len(read(runs_name + '/run{}/global{}_spTrain.traj'.format(i,i), index=':'))
        print('{}:'.format(i), traj_path)
        print('Ncalc:', Ncalc, 'Ebest:', Ebest)
        best_structures.append(a_best)
    except Exception as error:
        pass
    
write('temp_best_cur.traj', best_structures)
view(best_structures)
