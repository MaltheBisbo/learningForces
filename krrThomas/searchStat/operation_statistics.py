import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write

def calc_cummSuccess(success_indices, Nruns, run_length):
    cummSuccess = np.zeros(run_length)
    for index in success_indices:
        cummSuccess[index:] += 1
    cummSuccess /= Nruns
    return cummSuccess

def operation_statistics_singleTraj(traj, binsize=10, N=100, fig_name=None):
    operationStat = {}
    traj_length = len(traj)
    
    for i, a in enumerate(traj):
        operation = a.info['key_value_pairs']['origin']
        if operation in operationStat:
            operationStat[operation].append(i)
        else:
            operationStat[operation] = []
        
    fig, ax = plt.subplots()
    for key,value in operationStat.items():
        values_hist, bins = np.histogram(value, np.arange(0,traj_length+binsize, binsize))
        values_hist = np.array(values_hist)/binsize
        x = bins[:-1] + binsize/2
        ax.plot(x, values_hist, label=key)
    plt.legend()

    if fig_name is not None:
        plt.savefig(fig_name)

def operation_statistics(runs_path, binsize=10, N=100, fig_name=None, make_fig=False):
    operationStat = {}
    traj_list = []
    for i in range(N):
        try:
            traj = read(runs_path + 'run{}/global{}_spTrain.traj'.format(i,i), index=':')
            traj_list.append(traj)
        except:
            break
    Nruns = len(traj_list)
    max_runLength = np.max(np.array([len(traj) for traj in traj_list]))
    
    for traj in traj_list:
        for i, a in enumerate(traj):
            operation = a.info['key_value_pairs']['origin']
            if operation in operationStat:
                operationStat[operation].append(i)
            else:
                operationStat[operation] = []

    values_hist_list = []
    key_list = []
    for key,value in operationStat.items():
        values_hist, bins = np.histogram(value, np.arange(0,max_runLength+binsize, binsize))
        values_hist = np.array(values_hist)/(Nruns*binsize)
        values_hist_list.append(values_hist)
        key_list.append(key)
    x = bins[:-1] + binsize/2
    
    if make_fig:
        fig, ax = plt.subplots()
        for values_hist in values_hist_list:
            ax.plot(x, values_hist, label=key)
        plt.legend()

        if fig_name is not None:
            plt.savefig(fig_name)

    return x, values_hist_list, key_list

def operation_energy_corellation_singleTraj(traj, fig_name='operEnergy_corelation.pdf'):
    E = np.array([a.get_potential_energy() for a in traj])
    operationStat = {}
    for i, a in enumerate(traj):
        operation = a.info['key_value_pairs']['origin']
        if operation in operationStat:
            operationStat[operation].append(i)
        else:
            operationStat[operation] = []
        
    fig, ax = plt.subplots()
    for label, index in operationStat.items():
        ax.scatter(index, E[index], marker='.', label=label)
    plt.legend()

    if fig_name is not None:
        plt.savefig(fig_name)


def operation_energy_corellation(runs_path, N = 100, fig_name='operEnergy_corelation.pdf'):
    traj_list = []
    for i in range(N):
        try:
            traj = read(runs_path + 'run{}/global{}_spTrain.traj'.format(i,i), index=':')
            traj_list.append(traj)
        except:
            break
    Nruns = len(traj_list)
    Ncolumns = 1
    
    figsize_x = 5
    figsize_y = Nruns * 4
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    plt.subplots_adjust(left=1/figsize_x, right=1-1/figsize_x, wspace=7/figsize_x,
                        bottom=1/figsize_y, top=1-1/figsize_y, hspace=7/figsize_y)
    for i, traj in enumerate(traj_list):
        E = np.array([a.get_potential_energy() for a in traj])

        operationStat = {}
        for k, a in enumerate(traj):
            operation = a.info['key_value_pairs']['origin']
            if operation in operationStat:
                operationStat[operation].append(k)
            else:
                operationStat[operation] = []

        ax = fig.add_subplot(Nruns,Ncolumns,Ncolumns*i+1)
        for label, index in operationStat.items():
            ax.scatter(index, E[index], marker='.', label=label)
        plt.legend()

    if fig_name is not None:
        plt.savefig(fig_name)
