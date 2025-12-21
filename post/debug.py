import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data = pd.read_csv('debug.txt', sep=', ')

# plt.figure()
# plt.plot(data['cx'], data['cy'], 'o')
# plt.quiver(data['cx'], data['cy'], data['nx'], data['ny'])
# # put elem1 and elem2 text next to node cx, cy
# # for i in range(len(data)):
# #     if data['cx'][i] < 1 and data['cx'][i] > -1 and data['cy'][i] < 1 and data['cy'][i] > -1:
# #         plt.text(data['cx'][i], data['cy'][i], f"{data['elem1'][i]},{data['elem2'][i]}", fontsize=12)
# plt.xlabel('Centroid X')
# plt.ylabel('Centroid Y')
# plt.axis('equal')
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])

data = np.loadtxt('debug.txt')
plt.figure()
plt.plot(data[0,:], data[1,:], 'o')
plt.axis('equal')

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
