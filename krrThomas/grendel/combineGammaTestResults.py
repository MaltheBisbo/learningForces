import numpy as np
import matplotlib.pyplot as plt

Rc1 = 7
binwidth1 = 0.1
sigma1 = 0.4
gamma_array = np.linspace(1,24,24).astype(int)

results = []
for gamma in gamma_array:
    filename_load = 'results_radialFing/Rc7_gammaVaried/SnO_radialResults_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}_gamma{3:d}.txt'.format(Rc1, binwidth1, sigma1, gamma)
    result = np.loadtxt(filename_load, delimiter='\t')
    results.append(result)

results = np.array(results)
print(results)

#filename_save = 'results_radialFing/Rc7_gammaVaried/SnO_radialFeatures_gauss_Rc1_{0:d}_binwidth1_{1:.2f}_sigma1_{2:.1f}_gamma1-24.txt'.format(Rc1, binwidth1, sigma1)
#np.savetxt(filename_save, results, delimiter='\t')

MAE_array = results[:, -13:-10]

plt.figure()
plt.title('Radial fingerprint: Rcut=7, binwidth=0.1, sigma=0.4')
plt.plot(gamma_array, MAE_array[:,0], label='Ntrain=225')
plt.plot(gamma_array, MAE_array[:,1], label='Ntrain=356')
plt.plot(gamma_array, MAE_array[:,2], label='Ntrain=564')
plt.legend()
plt.xlabel('gamma')
plt.ylabel('MAE')
plt.show()

