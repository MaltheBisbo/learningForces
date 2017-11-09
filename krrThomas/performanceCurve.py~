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
dataML5 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax6/all_performance_MLenhanced.txt', delimiter='\t')
dataML5 = dataML5.reshape((200,7))
dataML5 = dataML5[~np.isnan(dataML5[:,0])]

NiterML5 = dataML5[:,0]
NfevML5 = dataML5[:,1]
EbestML5 = dataML5[:,2]

NiterML5_values, NiterML5_base = np.histogram(NiterML5, 1000)
NfevML5_values, NfevML5_base = np.histogram(NfevML5, 1000)

NiterML5_cum = np.cumsum(NiterML5_values)
NfevML5_cum = np.cumsum(NfevML5_values)

### MLenhanced - single target relax (Saving only relaxed)###
dataML6 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax7/all_performance_MLenhanced.txt', delimiter='\t')
dataML6 = dataML6.reshape((200,7))
dataML6 = dataML6[~np.isnan(dataML6[:,0])]

NiterML6 = dataML6[:,0]
NfevML6 = dataML6[:,1]
EbestML6 = dataML6[:,2]

NiterML6_values, NiterML6_base = np.histogram(NiterML6, 1000)
NfevML6_values, NfevML6_base = np.histogram(NfevML6, 1000)

NiterML6_cum = np.cumsum(NiterML6_values)
NfevML6_cum = np.cumsum(NfevML6_values)

plt.figure(1)
plt.title('Search performance - Search iterations')
plt.plot(Niter_base[:-1], Niter_cum/200, label='Without ML')
plt.plot(NiterML_base[:-1], NiterML_cum/200, label='With ML')
plt.plot(NiterML5_base[:-1], NiterML5_cum/200, label='With ML2')
plt.plot(NiterML6_base[:-1], NiterML6_cum/200, label='With ML3')
plt.legend()
plt.xlabel('# search iterations')
plt.ylabel('Succes rate')

plt.figure(2)
plt.title('Search performance - Function evaluations')
plt.plot(Nfev_base[:-1], Nfev_cum/200, label='Without ML')
plt.plot(NfevML_base[:-1], NfevML_cum/200, label='With ML')
plt.plot(NfevML5_base[:-1], NfevML5_cum/200, label='With ML2')
plt.plot(NfevML6_base[:-1], NfevML6_cum/200, label='With ML3')
plt.legend()
plt.xlabel('# function evaluations')
plt.ylabel('Succes rate')
plt.show()





