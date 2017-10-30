import numpy as np

data = np.zeros((200,3))
for i in range(200):
    data[i,:] = np.loadtxt('performance_MLenhanced' + str(i) + '.txt', delimiter='\t')

np.savetxt('all_performance_MLenhanced.txt', data, delimiter='\t')
