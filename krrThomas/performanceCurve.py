import numpy as np
import matplotlib.pyplot as plt

### BASECASE ###
data = np.loadtxt('grendel/basecase_data/all_performance_base.txt', delimiter='\t')
data = data.reshape((200,3))
data = data[~np.isnan(data[:,0])]

Niter = data[:,0]
Nfev = data[:,1]
Ebest = data[:,2]

Niter_values, Niter_base = np.histogram(Niter, 1000)
Nfev_values, Nfev_base = np.histogram(Nfev, 1000)

Niter_cum = np.cumsum(Niter_values)
Nfev_cum = np.cumsum(Nfev_values)

### MLenhanced ###
dataML = np.loadtxt('grendel/MLenhanced_data/all_performance_MLenhanced.txt', delimiter='\t')

NiterML = dataML[:,0]
NfevML = dataML[:,1]
EbestML = dataML[:,2]

NiterML_values, NiterML_base = np.histogram(NiterML, 1000)
NfevML_values, NfevML_base = np.histogram(NfevML, 1000)

NiterML_cum = np.cumsum(NiterML_values)
NfevML_cum = np.cumsum(NfevML_values)

### MLenhanced - single target relax (Saving relaxed + unrelaxed)###
dataML5 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax10/all_performance_MLenhanced.txt', delimiter='\t')
dataML5 = dataML5.reshape((200,8))
dataML5 = dataML5[~np.isnan(dataML5[:,0])]

NiterML5 = dataML5[:,0]
NfevML5 = dataML5[:,1]
EbestML5 = dataML5[:,2]
t_run5 = dataML5[:,-4]
t_relax5 = dataML5[:,-3]
t_train5 = dataML5[:,-2]
t_NerrorAbove5 = dataML5[:,-1]

print('t_run:', np.mean(t_run5))
print('t_relax:', np.mean(t_relax5))
print('t_train:', np.mean(t_train5))
print('t_NerrorAbove5:', np.mean(t_NerrorAbove5))

NiterML5_values, NiterML5_base = np.histogram(NiterML5, 1000)
NfevML5_values, NfevML5_base = np.histogram(NfevML5, 1000)

NiterML5_cum = np.cumsum(NiterML5_values)
NfevML5_cum = np.cumsum(NfevML5_values)

### MLenhanced - single target relax (Saving only relaxed)###
dataML6 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax10/all_performance_MLenhanced.txt', delimiter='\t')
dataML6 = dataML6.reshape((200,8))
dataML6 = dataML6[~np.isnan(dataML6[:,0])]

NiterML6 = dataML6[:,0]
NfevML6 = dataML6[:,1]
EbestML6 = dataML6[:,2]
t_run = dataML6[:,-3]
t_relax = dataML6[:,-2]
t_train = dataML6[:,-1]

print('t_run:', np.mean(t_run))
print('t_relax:', np.mean(t_relax))
print('t_train:', np.mean(t_train))

NiterML6_values, NiterML6_base = np.histogram(NiterML6, 1000)
NfevML6_values, NfevML6_base = np.histogram(NfevML6, 1000)

NiterML6_cum = np.cumsum(NiterML6_values)
NfevML6_cum = np.cumsum(NfevML6_values)

### MLenhanced - single target relax (Saving only relaxed)###
dataML7 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax13/all_performance_MLenhanced.txt', delimiter='\t')
dataML7 = dataML7.reshape((200,8))
dataML7 = dataML7[~np.isnan(dataML7[:,0])]

NiterML7 = dataML7[:,0]
NfevML7 = dataML7[:,1]
EbestML7 = dataML7[:,2]
t_run = dataML7[:,-4]
t_relax = dataML7[:,-3]
t_train = dataML7[:,-2]

print('t_run:', np.mean(t_run))
print('t_relax:', np.mean(t_relax))
print('t_train:', np.mean(t_train))

NiterML7_values, NiterML7_base = np.histogram(NiterML7, 1000)
NfevML7_values, NfevML7_base = np.histogram(NfevML7, 1000)

NiterML7_cum = np.cumsum(NiterML7_values)
NfevML7_cum = np.cumsum(NfevML7_values)

plt.figure(1)
plt.title('Search performance - Search iterations')
plt.plot(Niter_base[:-1], Niter_cum/200, label='Without ML')
plt.plot(NiterML_base[:-1], NiterML_cum/200, label='With ML')
plt.plot(NiterML5_base[:-1], NiterML5_cum/200, label='With ML2')
plt.plot(NiterML6_base[:-1], NiterML6_cum/200, label='With ML3')
plt.plot(NiterML7_base[:-1], NiterML7_cum/200, label='With ML4')
plt.legend()
plt.xlabel('# search iterations')
plt.ylabel('Succes rate')

plt.figure(2)
plt.title('Search performance - Function evaluations')
plt.plot(Nfev_base[:-1], Nfev_cum/200, label='Without ML')
plt.plot(NfevML_base[:-1], NfevML_cum/200, label='With ML')
plt.plot(NfevML5_base[:-1], NfevML5_cum/200, label='With ML2')
plt.plot(NfevML6_base[:-1], NfevML6_cum/200, label='With ML3')
plt.plot(NfevML7_base[:-1], NfevML7_cum/200, label='With ML4')
plt.legend()
plt.xlabel('# function evaluations')
plt.ylabel('Succes rate')
plt.show()





