#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

import sys

runs_name = sys.argv[1]

best_structures = []
for i in range(20):
    try:
        traj_path = runs_name + '/run{}/global{}_finalPop.traj'.format(i,i)
        a_best = read(traj_path, index='0')
        Ebest = a_best.get_potential_energy()
        print('{}:'.format(i), traj_path)
        print(Ebest)
        best_structures.append(a_best)
    except Exception as error:
        pass


view(best_structures)
