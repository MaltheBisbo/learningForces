import numpy as np

def extract(i):
    try:
        return np.loadtxt('performance_MLenhanced' + str(i) + '.txt', delimiter='\t')
    except OSError:
        print('No file')

data = np.zeros((88,7))
k = 0
for i in range(88):
    output = extract(i)
    if output is not None:
        data[k,:] = output
        k += 1
data = data[:k]
print(data)
print(data.shape[0])
np.savetxt('all_performance_MLenhanced.txt', data, delimiter='\t')
