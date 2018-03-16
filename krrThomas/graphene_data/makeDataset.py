import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

atoms = []
cand_number = 0
while cand_number < 420:
    try:
        a = read('cand{0:d}relax_lcao.traj'.format(cand_number), index=':')
        print(cand_number)
        cand_number += 1
    except IOError:
        cand_number += 1
        continue
    NrelaxSteps = len(a)
    atoms.append(a[0])
    atoms.append(a[int(NrelaxSteps/10)])
    atoms.append(a[int(NrelaxSteps/4)])
    atoms.append(a[-1])
write('graphene_all2.traj', atoms)
