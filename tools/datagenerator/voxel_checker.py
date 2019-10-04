import numpy as np
import os
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from math import cos,sin,pi
import random
import copy

random.seed(119)
seed = random.randint(1,200)
np.random.seed(seed)
names = ['t','intensity','id',
         'x','y','z',
         'azimuth','range','pid']

formats = ['int64', 'uint8', 'uint8',
           'float32', 'float32', 'float32',
           'float32', 'float32', 'int32']
path = "/home/orkeny/PycharmProjects/testProject/selected_obj/facade/"
szt_names = ['x','y','z']

szt_formats = ['float64', 'float64', 'float64', 'int64', 'int64', 'int64']

name = "0061.txt"

data_mx = np.loadtxt(path+name)
data_mx = data_mx[:, 0:3]
print(data_mx.dtype)
data = dict(zip(szt_names, data_mx.T))

# binType = np.dtype( dict(names=names, formats=formats))
temp = np.copy(data['y'])
data['y'] = np.copy(data['z'])
data['z'] = np.copy(temp)

# data = np.fromfile(path+name, binType)
#data['z']  = np.max(data['z']) - data['z']
data['z'] -= np.min(data['z'])
data['x'] -= 0.5*(np.min(data['x'])+np.max(data['x']))
data['y'] -= 0.5*(np.min(data['y'])+np.max(data['y']))

k = 1
rot_angles = np.random.randint(180, size=(1, k))
rot_angles = rot_angles.squeeze()
rot_angles = np.insert(rot_angles, 0, 0)
print(rot_angles)
for angle in rot_angles:

    rot_data = copy.deepcopy(data)
    f_angle = angle*pi / 180.0
    cnt = 0
    for x,y,z in zip(rot_data['x'].T, rot_data['y'].T, rot_data['z'].T):
        #rot_data['x'] = cos(f_angle) * rot_data['x'].T - sin(f_angle) * rot_data['y'].T
        #rot_data['y'] = sin(f_angle) * rot_data['x'].T + cos(f_angle) * rot_data['y'].T
        #rot_data['z'] = np.copy(data['z'])
        rot_data['x'][cnt] = cos(f_angle) * x - sin(f_angle) * y
        rot_data['y'][cnt] = sin(f_angle) * x + cos(f_angle) * y
        cnt += 1
    rot_data['x'] += 2.5 - 0.5 * (np.min(rot_data['x']) + np.max(rot_data['x']))
    rot_data['y'] += 2.5 - 0.5 * (np.min(rot_data['y']) + np.max(rot_data['y']))

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data['x'].T, data['y'].T, data['z'].T)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0,5)
    ax.set_ylim3d(0,5)
    ax.set_zlim3d(0,5)
    plt.title('point cloud')

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(rot_data['x'].T, rot_data['y'].T, rot_data['z'].T)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0,5)
    ax.set_ylim3d(0,5)
    ax.set_zlim3d(0,5)
    plt.title('Rotated point cloud: '+str(angle)+ 'Degree')
    plt.show()
