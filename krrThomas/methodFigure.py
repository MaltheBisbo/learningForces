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
    p1 = np.array(p1)
    p2 = np.array(p2)
    vec = p2 - p1
    diff = vec - (head_length + stop_before) * vec/np.linalg.norm(vec)
    plt.arrow(p1[0], p1[1], diff[0], diff[1], head_width=0.05, head_length=0.1, fc='k', ec='k')

fs = 15

plt.figure(figsize=(5,6))
plt.subplots_adjust(left=0.01, right=0.95,
                    bottom=0, top=1-1/14)
#plt.axis('equal')
#plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.axis([-1.7, 1.8, -1.6, 1.2])

# Plot functions
plt.plot(x_ML, f_ML)
plt.plot(x_true, f_true)

# Plot new point
plt.plot(1.4, func_ML(1.4), 'ro')

# Plot relaxation arrows
p1 = np.array([1.35, func_ML(1.35)+0.03])
p2 = np.array([1.0, func_ML(1.0)+0.03])
p3 = np.array([0.6, func_ML(0.6)+0.03])
p4 = np.array([0.3, func_ML(0.3)+0.03])
make_arrow(p1,p2, head_width=0.05, head_length=0.1, stop_before=0.03)
make_arrow(p2,p3, head_width=0.05, head_length=0.1, stop_before=0.03)
make_arrow(p3,p4, head_width=0.05, head_length=0.1, stop_before=0.03)

# Plot ML_relaxed point
plt.plot(0.3, func_ML(0.3), 'go')

# Plot arrow from ML to true potential
p = np.array([0.3, func_ML(0.3)-0.08])
q = np.array([0.3, func_true(0.3)+0.05])
make_arrow(p,q, head_width=0.05, head_length=0.1, stop_before=0.03)

# Plot ML_relaxed point on true surface
plt.plot(0.3, func_true(0.3), 'go')

# Plot text
plt.text(-0.35, -0.9, 'Single-\npoint', fontsize=fs)
plt.text(-1.6, 0.7, 'ML potential', fontsize=fs)
plt.text(-1.7, -0.49, 'True potential', fontsize=fs)
plt.text(0.6, 0.1, 'Relax', fontsize=fs)

p1 = np.array([1.4, 0.9])
p2 = np.array([1.4, 0.47])
make_arrow(p1,p2, head_width=0.05, head_length=0.1, stop_before=0.03)
plt.text(0.8, 1.0, 'New structure', fontsize=fs)

p1 = np.array([1.4, 0.9])
p2 = np.array([1.4, 0.47])
make_arrow(p1,p2, head_width=0.05, head_length=0.1, stop_before=0.03)
plt.text(0.8, 1.0, 'New structure', fontsize=fs)

make_arrow([0.9, -1.38],[0.4, -1.38], head_width=0.05, head_length=0.1, stop_before=0.03)
plt.text(1.0, -1.5, 'Use for\ntraining', fontsize=fs)

plt.savefig('MLrelaxFigure.pdf', format='pdf')
plt.show()

