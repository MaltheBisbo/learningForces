import numpy as np
import matplotlib.pyplot as plt

def f_cutoff(r, gamma, Rc):
    """
    Polinomial cutoff function in the, with the steepness determined by "gamma"
    gamma = 2 resembels the cosine cutoff function.
    For large gamma, the function goes towards a step function at Rc.
    """
    return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma

def surface_area(r):
    """
    """
    return 4*np.pi*(r**2)

Rc = 5
r = np.linspace(0.5, Rc, 100)

gamma = 3
func1 = f_cutoff(r, gamma, Rc) / surface_area(r)
func2 = f_cutoff(r, gamma, Rc)
func3 = 1 / surface_area(r)

plt.figure(1)
plt.plot(r, func1, label='cutoff/surface_area')
plt.plot(r, func2, label='cutoff')
plt.plot(r, func3, label='1/surface_area')
plt.legend()
plt.show()
