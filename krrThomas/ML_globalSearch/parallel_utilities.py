import numpy as np
from ase import Atoms


def copy_atoms(a):
    acopy = Atoms(symbols=a.get_chemical_symbols(),
                  cell=a.cell,
                  pbc=a.pbc,
                  calculator=a.get_calculator)
    acopy.info['key_value_pairs'] = {}
    return acopy


def sync_atoms(comm, atoms_list, Epred_list=None, error_list=None, dmin_list=None, kappa=None, operation_dict=None, operation_index=None):
    Neach = len(atoms_list)
    Natoms = atoms_list[0].positions.shape[0]

    # Wrap atoms into cell
    for a in atoms_list: a.wrap()
    
    # Destribute all data among all communicators
    pos_list = np.array([a.positions for a in atoms_list])
    pos_all = np.empty(Neach*comm.size * 3*Natoms, dtype=float)
    comm.all_gather(pos_list.reshape(-1), pos_all)
    pos_all = pos_all.reshape((Neach*comm.size, Natoms, 3))

    if Epred_list is not None:
        E_all = np.empty(Neach * comm.size, dtype=float)
        comm.all_gather(Epred_list, E_all)
    if error_list is not None:
        error_all = np.empty(Neach * comm.size, dtype=float)
        comm.all_gather(error_list, error_all)
    if dmin_list is not None:
        dmin_all = np.empty(Neach * comm.size, dtype=float)
        comm.all_gather(dmin_list, dmin_all)
    if operation_dict is not None and operation_index is not None:
        operation_index_all = np.empty(Neach * comm.size, dtype=int)
        comm.all_gather(operation_index, operation_index_all)

    atoms_all=[]
    for i, pos in enumerate(pos_all):
        a = copy_atoms(atoms_list[0])
        a.set_positions(pos)
        a.set_constraint(atoms_list[0].constraints)
        if Epred_list is not None:
            a.info['key_value_pairs']['predictedEnergy'] = E_all[i]
        if error_list is not None:
            a.info['key_value_pairs']['predictedError'] = error_all[i]
        if dmin_list is not None:
            a.info['key_value_pairs']['dmin'] = dmin_all[i]
        if kappa is not None:
            a.info['key_value_pairs']['fitness'] = E_all[i] - kappa*error_all[i]
        if operation_dict is not None and operation_index is not None:
            a.info['key_value_pairs']['origin'] = operation_dict[operation_index_all[i]]
        atoms_all.append(a)

    return atoms_all
