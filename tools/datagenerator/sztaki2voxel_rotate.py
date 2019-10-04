# -*- coding: utf-8 -*-
""" Kiír minden adatok adott mappában további feldolgozásra NUMPY tömbökbe pickle használatával"""
import numpy as np
import os
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from math import cos,sin,pi
import random
import copy

random.seed(19)
seed = random.randint(1,200)
np.random.seed(seed)
names = ['t','intensity','id',
         'x','y','z',
         'azimuth','range','pid']

formats = ['int64', 'uint8', 'uint8',
           'float32', 'float32', 'float32',
           'float32', 'float32', 'int32']

labels = []
inputs = []
mypath = "/home/orkeny/PycharmProjects/testProject/selected_obj/sztaki_furn/"
instances = [files for files in listdir(mypath)]
cnt = 0
szt_names = ['x','y','z']

for inst in instances:
    if inst.endswith('.txt'):
        #binType = np.dtype( dict(names=names, formats=formats))
        #data = np.fromfile(mypath+inst, binType)
        data_mx = np.loadtxt(mypath+inst)
        data_mx = data_mx[:, 0:3]
        print(data_mx.dtype)
        data = dict(zip(szt_names, data_mx.T))

        # binType = np.dtype( dict(names=names, formats=formats))
        temp = data['y']
        data['y'] = data['z']
        data['z'] = temp

        label = 'szt_furn'
        range_x = range_y = range_z = 40


        # Centering
        #data['z'] = np.max(data['z']) - data['z']
        data['z'] -= np.min(data['z'])
        data['x'] -= 0.5 * (np.min(data['x']) + np.max(data['x']))
        data['y'] -= 0.5 * (np.min(data['y']) + np.max(data['y']))


        # Data augmentation
        # Tükrözés
        #data['y'] = np.max(data['y']) - data['y']
        #data['x'] = np.max(data['x']) - data['x']

        # Forgatás
        k = 20
        rot_angles = np.random.randint(180, size=(1, k))
        rot_angles = rot_angles.squeeze()
        rot_angles = np.insert(rot_angles, 0, 0)
        print(rot_angles)
        for angle in rot_angles:
            voxel_data = np.zeros((range_x, range_y, range_z), dtype='uint8')

            rot_data = copy.deepcopy(data)
            f_angle = angle * pi / 180.0
            idx = 0
            for x, y, z in zip(rot_data['x'].T, rot_data['y'].T, rot_data['z'].T):
                # rot_data['x'] = cos(f_angle) * rot_data['x'].T - sin(f_angle) * rot_data['y'].T
                # rot_data['y'] = sin(f_angle) * rot_data['x'].T + cos(f_angle) * rot_data['y'].T
                # rot_data['z'] = np.copy(data['z'])
                rot_data['x'][idx] = cos(f_angle) * x - sin(f_angle) * y
                rot_data['y'][idx] = sin(f_angle) * x + cos(f_angle) * y
                idx += 1
            rot_data['x'] += 2.5 - 0.5 * (np.min(rot_data['x']) + np.max(rot_data['x']))
            rot_data['y'] += 2.5 - 0.5 * (np.min(rot_data['y']) + np.max(rot_data['y']))


            res_x = res_y = res_z = 5.0/ range_x

            for x,y,z in np.vstack([rot_data['x'], rot_data['y'], rot_data['z']]).T:
                x1 = int(x / res_x)
                y1 = int(y / res_y)
                z1 = int(z / res_z)
                if x1 < range_x and x1 > -1 and y1 < range_y and y1 > -1 and z1 < range_z and z1 > -1:
                    voxel_data[x1][y1][z1] = 1

            with open('rotation/'+label+'/szt_'+str(cnt)+'.pkl', "wb") as f:
                pickle.dump(voxel_data, f)
                cnt += 1
"""""
            # Kirajzolás
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(rot_data['x'].T, rot_data['y'].T, rot_data['z'].T)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim3d(0, 5)
            ax.set_ylim3d(0, 5)
            ax.set_zlim3d(0, 5)
            plt.title(label+' point cloud')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.voxels(voxel_data, edgecolors='gray')

            ax2.set_xlabel('X Label')
            ax2.set_ylabel('Y Label')
            ax2.set_zlabel('Z Label')
            ax2.set_xlim3d(0, 40)
            ax2.set_ylim3d(0, 40)
            ax2.set_zlim3d(0, 40)
            plt.title(label+' voxels')
            plt.show()

            print(np.min(data['x']))
            print(np.max(data['x']))
            print(np.min(data['y']))
            print(np.max(data['y']))
            print(np.min(data['z']))
            print(np.max(data['z']))


"""