import numpy as np
import matplotlib.pyplot as plt

x_ML_offset = 0.3
def func_ML(x):
    offset = x_ML_offset
    f_ML = - 1.6*np.exp(-(x-offset)**2/2) + 1.3 
    return f_ML

def func_true(x):
    f_true = -1.5*np.exp(-x**2)
    return f_true


x_ML = np.linspace(-1.3,1.3,100) + x_ML_offset
x_true = np.linspace(-1,1.7,100)
f_ML = func_ML(x_ML)
f_true = func_true(x_true)

def make_arrow(p1,p2, head_width, head_length, stop_before):
    vec = p2 - p1
    diff = vec - (head_length + stop_before) * vec/np.linalg.norm(vec)
    plt.arrow(p1[0], p1[1], diff[0], diff[1], head_width=0.05, head_length=0.1, fc='k', ec='k')


plt.figure()
plt.axis('equal')
plt.axis([-2, 2.5, -2, 1.5])

# Plot functions
plt.plot(x_ML, f_ML)
plt.plot(x_true, f_true)

# Plot new point
plt.scatter(1.4, func_ML(1.4), color='r')

# Plot relaxation arrows
p1 = np.array([1.35, func_ML(1.35)+0.03])
p2 = np.array([1.0, func_ML(1.0)+0.03])
p3 = np.array([0.6, func_ML(0.6)+0.03])
p4 = np.array([0.3, func_ML(0.3)+0.03])
make_arrow(p1,p2, head_width=0.05, head_length=0.1, stop_before=0.03)
make_arrow(p2,p3, head_width=0.05, head_length=0.1, stop_before=0.03)
make_arrow(p3,p4, head_width=0.05, head_length=0.1, stop_before=0.03)

# Plot ML_relaxed point
plt.scatter(0.3, func_ML(0.3), color='g')

# Plot arrow from ML to true potential
p = np.array([0.3, func_ML(0.3)-0.08])
q = np.array([0.3, func_true(0.3)+0.05])
make_arrow(p,q, head_width=0.05, head_length=0.1, stop_before=0.03)

# Plot ML_relaxed point on true surface
plt.scatter(0.3, func_true(0.3), color='g')

# Plot text
plt.text(-0.35, -0.9, 'Calculate\ntrue\nenergy')
plt.text(-1.6, 0.7, 'ML potential')
plt.text(-1.6, -0.49, 'True potential')
plt.text(0.6, 0.1, 'Relax')

p1 = np.array([1.4, 0.9])
p2 = np.array([1.4, 0.47])
make_arrow(p1,p2, head_width=0.05, head_length=0.1, stop_before=0.03)
plt.text(1.1, 1.0, 'New structure')

plt.show()

