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

### MLenhanced - single target relax ###
dataML2 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax/all_performance_MLenhanced.txt', delimiter='\t')
dataML2 = dataML2.reshape((200,3))
dataML2 = dataML2[~np.isnan(dataML2[:,0])]

NiterML2 = dataML2[:,0]
NfevML2 = dataML2[:,1]
EbestML2 = dataML2[:,2]

NiterML2_values, NiterML2_base = np.histogram(NiterML2, 1000)
NfevML2_values, NfevML2_base = np.histogram(NfevML2, 1000)

NiterML2_cum = np.cumsum(NiterML2_values)
NfevML2_cum = np.cumsum(NfevML2_values)

plt.figure(1)
plt.title('Search performance - Search iterations')
plt.plot(Niter_base[:-1], Niter_cum/200, label='Without ML')
plt.plot(NiterML_base[:-1], NiterML_cum/200, label='With ML')
plt.plot(NiterML2_base[:-1], NiterML2_cum/200, label='With ML2')
plt.legend()
plt.xlabel('# search iterations')
plt.ylabel('Succes rate')

plt.figure(2)
plt.title('Search performance - Function evaluations')
plt.plot(Nfev_base[:-1], Nfev_cum/200, label='Without ML')
plt.plot(NfevML_base[:-1], NfevML_cum/200, label='With ML')
plt.plot(NfevML2_base[:-1], NfevML2_cum/200, label='With ML2')
plt.legend()
plt.xlabel('# function evaluations')
plt.ylabel('Succes rate')
plt.show()





