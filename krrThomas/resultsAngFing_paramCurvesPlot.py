import numpy as np
import matplotlib.pyplot as plt

eta_array = np.linspace(1, 30, 15).astype(int)
results = np.loadtxt('resultsAngFing_paramCurves2.txt', delimiter='\t')
label_array = []

for name in ['', '_r_fcut', '_fcut']:
    for sigma2 in [0.05, 0.1, 0.2]:
        label_array.append('{0:s} sigmaAng={1:.2f}'.format(name, sigma2))

plt.figure()
plt.plot(eta_array, results.T)
plt.show()
