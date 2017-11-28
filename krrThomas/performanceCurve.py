import numpy as np
import matplotlib.pyplot as plt

### BASECASE ###
data = np.loadtxt('grendel/dataN19_basecase/all_performance_base.txt', delimiter='\t')
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
dataML = np.loadtxt('grendel/dataN19_ML_multipleTargetRelax/all_performance_MLenhanced.txt', delimiter='\t')

NiterML = dataML[:,0]
NfevML = dataML[:,1]
EbestML = dataML[:,2]

NiterML_values, NiterML_base = np.histogram(NiterML, 1000)
NfevML_values, NfevML_base = np.histogram(NfevML, 1000)

NiterML_cum = np.cumsum(NiterML_values)
NfevML_cum = np.cumsum(NfevML_values)

### MLenhanced - single target relax (Saving relaxed + unrelaxed) ###
### {r_min, r_max} structure-validity check ###
dataML2 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax2/all_performance_MLenhanced.txt', delimiter='\t')
dataML2 = dataML2.reshape((200,7))
dataML2 = dataML2[~np.isnan(dataML2[:,0])]

NiterML2 = dataML2[:,0]
NfevML2 = dataML2[:,1]
EbestML2 = dataML2[:,2]
t_run2 = dataML2[:,-3]
t_relax2 = dataML2[:,-2]
t_train2 = dataML2[:,-1]

print('t_run:', np.mean(t_run2))
print('t_relax:', np.mean(t_relax2))
print('t_train:', np.mean(t_train2))

NiterML2_values, NiterML2_base = np.histogram(NiterML2, 1000)
NfevML2_values, NfevML2_base = np.histogram(NfevML2, 1000)

NiterML2_cum = np.cumsum(NiterML2_values)
NfevML2_cum = np.cumsum(NfevML2_values)

### MLenhanced - single target relax (Saving only relaxed) ###
### E<0 structure-validity check ###
### No new perturbation when E>0 ###
dataML3 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax3/all_performance_MLenhanced.txt', delimiter='\t')
dataML3 = dataML3.reshape((200,8))
dataML3 = dataML3[~np.isnan(dataML3[:,0])]

NiterML3 = dataML3[:,0]
NfevML3 = dataML3[:,1]
EbestML3 = dataML3[:,2]
t_run = dataML3[:,-4]
t_relax = dataML3[:,-3]
t_train = dataML3[:,-2]

print('t_run:', np.mean(t_run))
print('t_relax:', np.mean(t_relax))
print('t_train:', np.mean(t_train))

NiterML3_values, NiterML3_base = np.histogram(NiterML3, 1000)
NfevML3_values, NfevML3_base = np.histogram(NfevML3, 1000)

NiterML3_cum = np.cumsum(NiterML3_values)
NfevML3_cum = np.cumsum(NfevML3_values)

### MLenhanced - single target relax (Saving only relaxed) ###
### err/sqrt(theta0) < 0.01 structure-validity check ###                                                                             
dataML4 = np.loadtxt('grendel/dataN19_ML_singleTargetRelax4/all_performance_MLenhanced.txt', delimiter='\t')
dataML4 = dataML4.reshape((200,8))
dataML4 = dataML4[~np.isnan(dataML4[:,0])]

NiterML4 = dataML4[:,0]
NfevML4 = dataML4[:,1]
EbestML4 = dataML4[:,2]
t_run = dataML4[:,-4]
t_relax = dataML4[:,-3]
t_train = dataML4[:,-2]

print('t_run:', np.mean(t_run))
print('t_relax:', np.mean(t_relax))
print('t_train:', np.mean(t_train))

NiterML4_values, NiterML4_base = np.histogram(NiterML4, 1000)
NfevML4_values, NfevML4_base = np.histogram(NfevML4, 1000)

NiterML4_cum = np.cumsum(NiterML4_values)
NfevML4_cum = np.cumsum(NfevML4_values)

plt.figure(1)
plt.title('Search performance - Search iterations')
plt.plot(Niter_base[:-1], Niter_cum/200, label='Without ML')
plt.plot(NiterML_base[:-1], NiterML_cum/200, label='With ML 1')
#plt.plot(NiterML2_base[:-1], NiterML2_cum/200, label='With ML2')
#plt.plot(NiterML3_base[:-1], NiterML3_cum/200, label='With ML3')
plt.plot(NiterML4_base[:-1], NiterML4_cum/200, label='With ML 2')
plt.legend()
plt.xlabel('# search iterations')
plt.ylabel('Succes rate')

plt.figure(2)
plt.title('Search performance - Function evaluations')
plt.plot(Nfev_base[:-1], Nfev_cum/200, label='Without ML')
plt.plot(NfevML_base[:-1], NfevML_cum/200, label='With ML 1')
#plt.plot(NfevML2_base[:-1], NfevML2_cum/200, label='With ML2')
#plt.plot(NfevML3_base[:-1], NfevML3_cum/200, label='With ML3')
plt.plot(NfevML4_base[:-1], NfevML4_cum/200, label='With ML 2')
plt.legend()
plt.xlabel('# function evaluations')
plt.ylabel('Succes rate')
plt.show()





