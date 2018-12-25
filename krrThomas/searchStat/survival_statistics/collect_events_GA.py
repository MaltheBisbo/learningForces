########################################################################################
#
# This script will create an event.npy file to be used with the survival_stats.py
# script. It is designed to work with some of the run EA programs by matjoerg.
#
########################################################################################


import os
import sys
import numpy as np
from ase.io import read

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

# read through all folders
ref = read(os.path.join(cwd,'..','gm.traj')).get_potential_energy()
time = []
event = []

for f in os.listdir(cwd):
    f2 = os.path.join(cwd, f)
    if not os.path.isdir(f2) or 'run' not in f:
        continue
    print(f)
    ftraj = os.path.join(f2, 'all_candidates.traj')
    if not os.path.isfile(ftraj):
        print('No all_candidates.traj')
        continue
    try:
        ats = read(ftraj + '@:')
    except:
        ats = False
    if not ats:
        print('all_candidates.traj is empty')
        continue
    ats.sort(key = lambda x: x.info['confid'])
    found = False
    for i, a in enumerate(ats):
        if a.get_potential_energy() < ref + 0.025:
            time.append(i)
            found = True
            print('Best Structure Found After {} Attempts'.format(i))
            break
    if found:
        event.append(1)
    else:
        time.append(i)
        event.append(0)
        print('Best Structure Not Found')

np.save(os.path.join(cwd,'events.npy'),(time,event))

print('\nRuns:              {:5}'.format(len(time)))
print('Successful Runs:   {:5}'.format(sum(event)))
print('Success Rate:      {:5.0%}\n'.format(sum(event)/float(len(time))))

