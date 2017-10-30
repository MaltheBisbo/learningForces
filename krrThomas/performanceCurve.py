import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('grendel/basecase_data/all_performance_base.txt', delimiter='\t')
data = data.reshape((200,3))
data = data[~np.isnan(data[:,0])]

Niter = data[:,0]
Nfev = data[:,1]
Ebest = data[:,2]

Niter_values, base = np.histogram(Niter, bins=np.arange(201))
Nfev_values, base = np.histogram(Nfev, 200)

Niter_cum = np.cumsum(Niter_values)
Nfev_cum = np.cumsum(Nfev_values)

plt.figure(1)
plt.title('Search performance - iterations (without ML)')
plt.plot(np.arange(200)+0.5, Niter_cum/200)
plt.xlabel('# search iterations until global minimum')
plt.ylabel('Fraction of searches finging the global minimum')

plt.figure(2)
plt.title('Search performance - function evaluations (without ML)')
plt.plot(np.arange(200)+0.5, Nfev_cum)
plt.xlabel('# function evaluations until global minimum')
plt.ylabel('Fraction of searches finging the global minimum')
plt.show()
