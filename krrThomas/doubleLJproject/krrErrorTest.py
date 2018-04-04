import numpy as np
import matplotlib.pyplot as plt

def extract(i):
    try:
        return np.loadtxt('grendel/dataN19_ML_singleTargetRelax3/relaxedEnergiesML' + str(i) + '.txt', delimiter=' ')
    except OSError:
        print('No file')


Ndata = 200
Niter = 399
data = np.zeros((Ndata, Niter, 5))
k = 0
for i in range(Ndata):
    print(k)
    output = extract(i)[1:]
    if output is not None:
        data[k,:] = output
        k += 1

ErelML = data[:,:,0]
ErelTrue = data[:,:,1]
error = data[:,:,2]
theta = data[:,:,3]

dE = np.abs(ErelML - ErelTrue)

frac_fail = np.sum((ErelTrue > 0).astype(int))/(Ndata*Niter)
print('frac_fail:', frac_fail)

plt.figure(1)
plt.yscale('log')
plt.scatter(error, dE)
plt.scatter(error[ErelTrue > 0], dE[ErelTrue > 0], color='r')

filter2 = ErelTrue > 0
plt.figure(2)
plt.yscale('log')
plt.xscale('log')
plt.scatter(error, dE, alpha=0.2)
plt.scatter(error[filter2], dE[filter2], color='r', alpha=0.2)
plt.xlabel('krr_error')
plt.ylabel('abs(Epredict - Etarget)')

filter3 = ErelTrue > 0
plt.figure(3)
plt.yscale('log')
plt.xscale('log')
plt.scatter(error/np.sqrt(theta), dE, alpha=0.2)
plt.scatter(error[filter3]/np.sqrt(theta[filter3]), dE[filter3], color='r', alpha=0.2)
plt.xlabel('krr_error/sqrt(theta0)')
plt.ylabel('abs(Epredict - Etarget)')
plt.show()

plt.figure(4)
