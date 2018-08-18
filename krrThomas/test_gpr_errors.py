import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from gaussComparator import gaussComparator
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from delta_functions_multi.delta import delta as deltaFunc
from krr_ase import krr_class

atoms_all = read('SnO_data_all/all_-50.traj', index=':')
atoms_train = atoms_all[:500]
atoms_test = atoms_all[500:800]
E_train = np.array([a.get_potential_energy() for a in atoms_train])
E_test = np.array([a.get_potential_energy() for a in atoms_test])
a0 = atoms_all[0]
Rc1 = 6
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2

gamma = 2
eta = 20
use_angular = True


featureCalculator = Angular_Fingerprint(a0, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)

# Set up KRR-model
comparator = gaussComparator(featureCalculator=featureCalculator)
#delta_function = deltaFunc(cov_dist=2*covalent_radii[6])
delta_function = deltaFunc(atoms=a0, rcut=6)
krr = krr_class(comparator=comparator,
                featureCalculator=featureCalculator,
                delta_function=delta_function)





sigma_list = [10]  # np.logspace(0,2,10)

MAE_list = []
for sigma in sigma_list:
    pred_error_list = []
    Epred_list = []
    GSkwargs = {'sigma': [sigma], 'reg': [1e-5]}
    MAE, params = krr.train(atoms_train, k=3, **GSkwargs)
    for a in atoms_test:
        Epred, pred_error, _ = krr.predict_energy(a, return_error=True)
        Epred_list.append(Epred)
        pred_error_list.append(pred_error)
    Epred_list = np.array(Epred_list)
    MAE_list.append(np.mean(np.abs(Epred_list-E_test)))

    pred_error_list = np.array(pred_error_list)



bins = np.linspace(0,2.5,30)

test_error = np.abs(E_test-Epred_list)
plt.figure()
#plt.plot(sigma_list, MAE_list)
plt.hist(pred_error_list, bins, histtype='step', label='pred error')
plt.hist(test_error, bins, histtype='step', label='test error')
plt.hist(np.abs(test_error-pred_error_list), bins, histtype='step', label='test error')
plt.legend()
plt.show()
