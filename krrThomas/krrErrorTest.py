import numpy as np
import matplotlib.pyplot as plt

def extract(i):
    try:
        return np.loadtxt('grendel/dataN19_ML_singleTargetRelax13/relaxedEnergiesML' + str(i) + '.txt', delimiter=' ')
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

plt.figure(2)
plt.yscale('log')
plt.xscale('log')
plt.scatter(error, dE)
plt.scatter(error[error > 1], dE[error > 1], color='r')

filter3 = ErelTrue > -30
plt.figure(3)
plt.yscale('log')
plt.xscale('log')
plt.scatter(error/np.sqrt(theta), dE)
plt.scatter(error[filter3]/np.sqrt(theta[filter3]), dE[filter3], color='r')
plt.show()

plt.figure(4)
