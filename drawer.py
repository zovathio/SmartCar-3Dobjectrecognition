import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

with open("dataset/car/x_7.pkl","rb") as f:
    voxel_data = pickle.load(f)

voxel_data = voxel_data.reshape((40,40,40))

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_data, edgecolors='gray')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title(' voxels')
plt.show()




