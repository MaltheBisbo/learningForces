import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write

def errorEvolution(traj, kappa=1, fig_name=None):
    E = np.array([a.get_potential_energy() for a in traj])
    Epred = np.array([a.info['key_value_pairs']['predictedEnergy'] for a in traj])
    error_pred = np.array([a.info['key_value_pairs']['predictedError'] for a in traj])
    error_true = np.abs(E - Epred)
    
    fig, ax = plt.subplots()
    n = np.arange(len(traj))
    ax.semilogy(n, kappa*error_pred, label='kappa*error_pred')
    ax.semilogy(n, error_true, label='true_true')
    plt.legend()

    if fig_name is not None:
        plt.savefig(fig_name)

def errorEvolution_all(traj, kappa=1, fig_name=None):
    E = np.array([a.get_potential_energy() for a in traj])
    Epred = np.array([a.info['key_value_pairs']['predictedEnergy'] for a in traj])
    error_pred = np.array([a.info['key_value_pairs']['predictedError'] for a in traj])
    error_true = np.abs(E - Epred)
    
    fig, ax = plt.subplots()
    n = np.arange(len(traj))
    ax.semilogy(n, kappa*error_pred, label='kappa*error_pred')
    ax.semilogy(n, error_true, label='true_true')
    plt.legend()

    if fig_name is not None:
        plt.savefig(fig_name)


def plot_energy_and_error_batch(traj, binsize=1, kappa=1, with_true=False):
    Epred = np.array([a.info['key_value_pairs']['predictedEnergy'] for a in traj])
    error_pred = np.array([a.info['key_value_pairs']['predictedError'] for a in traj])

    if with_true:
        Etrue = np.array([a.get_potential_energy() for a in traj])
        error_true = np.abs(Etrue - Epred)

    if binsize > 1:
        Epred = np.mean(Epred.reshape(-1,binsize), axis=1)
        error_pred = np.mean(error_pred.reshape(-1,binsize), axis=1)
        if with_true:
            Etrue = np.mean(Etrue.reshape(-1,binsize), axis=1)
            error_true = np.mean(error_true.reshape(-1,binsize), axis=1)
        
    n = np.arange(len(Epred))
    fig, ax1 = plt.subplots()
    ax1.plot(n, Epred, color='steelblue', label='predicted energy')
    ax1.set_xlabel('singlepoint calculations')
    ax1.set_ylabel('Energy')
    
    ax2 = ax1.twinx()
    ax2.plot(n, kappa*error_pred, color='crimson', label='predicted error')
    ax2.set_ylabel('Error on energy')
    
    if with_true:
        ax1.plot(n, Etrue, linestyle='--', color='steelblue', label='true energy')
        ax2.plot(n, error_true, linestyle='--', color='crimson', label='true error')
    plt.legend()


def plot_fitness_composition_batch(run_path, binsize=1):
    candidates_all = []
    for i in range(400):
        print(i)
        try:
            candidates_i = read(run_path + '/all_MLcandidates/ML_relaxed{}.traj'.format(i), index=':')
            candidates_all.append(candidates_i)
        except Exception as e:
            print(e)
            break
    Niter = len(candidates_all)
    Ncand = len(candidates_all[0])
    
    fitness_all = np.zeros((Niter, Ncand))
    energy_all = np.zeros((Niter, Ncand))
    for i, traj in enumerate(candidates_all):
        for j, a in enumerate(traj):
            fitness_all[i,j] = a.info['key_value_pairs']['fitness']
            energy_all[i,j] = a.info['key_value_pairs']['predictedEnergy']
    
    fitness_best = np.min(fitness_all, axis=1)
    energy_best = np.min(energy_all, axis=1)
    error_best = energy_best - fitness_best

    fitness_worst = np.max(fitness_all, axis=1)
    energy_worst = np.max(energy_all, axis=1)
    
    d_energy_best = energy_best - energy_worst
    d_fitness_best = np.abs(d_energy_best) + error_best
    
    d_energy_best_frac = np.abs(d_energy_best)/d_fitness_best
    error_best_frac = error_best/d_fitness_best
    
    
    fitness_span = fitness_worst - fitness_best

    n = np.arange(len(fitness_all))
    fig, ax1 = plt.subplots()
    ax1.plot(n, d_energy_best_frac, color='steelblue', label='energy contribution')
    ax1.plot(n, error_best_frac, color='crimson', label='error contribution')
    ax1.set_xlabel('singlepoint calculations')
    ax1.set_ylabel('fractional contribution to fitness')
    
    
    ax2 = ax1.twinx()
    ax2.semilogy(n, d_fitness_best, color='k', label='fitness_span')
    ax2.set_ylabel('variation in fitness')

    plots1, labels1 = ax1.get_legend_handles_labels()
    plots2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(plots1 + plots2, labels1 + labels2)

if __name__ == '__main__':
    from ase.io import read

    i = 1
    traj = read('/home/mkb/DFTB/TiO_2layer/all_runs/runs14/run{}/global{}_spTrain.traj'.format(i,i), index=':100')

    #plot_energy_and_error_batch(traj, binsize=10, with_true=True)
    #errorEvolution(traj, kappa=2)

    run_path = '/home/mkb/DFTB/TiO_2layer/all_runs/runs14/run4'
    plot_fitness_composition_batch(run_path)
    plt.show()
