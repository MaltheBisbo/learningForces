import numpy as np
import matplotlib.pyplot as plt

def f_cutoff(r, gamma, Rc):
    """
    Polinomial cutoff function in the, with the steepness determined by "gamma"
    gamma = 2 resembels the cosine cutoff function.
    For large gamma, the function goes towards a step function at Rc.
    """
    if not gamma == 0:
        return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma
    else:
        return 1

Rc = 1
r_array = np.linspace(0,1,1000)
gamma_array = np.array([1,2,3,5])
plt.figure()
plt.title('Cutoff-function')
for gamma in gamma_array:
    fcut = np.array([f_cutoff(r,gamma,Rc) for r in r_array])
    plt.plot(r_array, fcut, label='gamma={}'.format(gamma))
plt.legend()
plt.xlabel('r/Rc')
plt.ylabel('fcut')
plt.show()
