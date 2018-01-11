import numpy as np
import matplotlib.pyplot as plt

eta_array = np.linspace(1, 30, 15).astype(int)
results = np.loadtxt('resultsAngFing_paramCurves2.txt', delimiter='\t')
label_array = []

k = 0
plt.figure()
for name in ['r2_fcut', 'r_fcut', 'fcut']:
    for sigma2 in [0.05, 0.1, 0.2]:
        label = '{0:s}, sig={1:.2f}'.format(name, sigma2)
        print(label)
        plt.plot(eta_array, results[k], label=label)
        k += 1

plt.title('MAE error as function of the emphasis on the angular part of the feature\n(for different normalizations and gauss widths)')
plt.xlabel('eta ("Amplitude" of angular part of feature)')
plt.ylabel('MAE')
plt.legend()
plt.show()
