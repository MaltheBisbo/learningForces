import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

im = plt.imread(get_sample_data('grace_hopper.jpg'))
ax.plot(range(10))
newax = fig.add_axes([0.1, 0.7, 0.1, 0.1], projection='polar')
