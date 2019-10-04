import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
import copy

random.seed(119)
seed = random.randint(1,200)

mypath = "/home/orkeny/PycharmProjects/testProject/dataset/"
folders = [files for files in listdir(mypath)]
cnt = 0
for label in folders:
    cpath = mypath+label+'/'
    instances = [files for files in listdir(cpath)]
    k = 5
    lim = 10
    if label == 'p':
        lim = 30
    elif label == 'c':
        lim = 30
    tr_x = np.random.randint(-lim, lim, size=(1, k))
    tr_x = tr_x.squeeze()
    tr_x = np.insert(tr_x, 0, 0)
    for inst in instances:
        if inst.endswith('.pkl'):
            with open(cpath+inst,"rb") as f:
                voxel_data = pickle.load(f)

            voxel_data = voxel_data.reshape((40,40,40))
            for tr in tr_x:
                tr_data_y = np.roll(voxel_data, tr, axis=1)
                with open('dataset/'+label+'/y_'+inst, "wb") as f:
                    pickle.dump(tr_data_y, f)
                    cnt += 1
"""
                fig = plt.figure()

                ax = fig.add_subplot(121, projection='3d')
                ax.voxels(voxel_data, edgecolors='gray')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.voxels(tr_data_x, edgecolors='gray')
                ax2.set_xlabel('X Label')
                ax2.set_ylabel('Y Label')
                ax2.set_zlabel('Z Label')
                plt.title(label+str(tr))
                plt.show()
"""





