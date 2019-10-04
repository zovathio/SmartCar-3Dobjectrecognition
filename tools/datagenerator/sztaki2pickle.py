import pickle
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mypath="/home/orkeny/PycharmProjects/3Dobject_class/test"
instances = [files for files in listdir(mypath+"/")]
i = 0
for inst in instances:
    print(i)
    f = open(mypath+"/"+inst, "rb")
    f.seek(os.SEEK_SET)
    data = np.fromfile(f, dtype=np.int32)
    f.close()
    print(data.max())
    data = data.reshape((40,40,40))
    print(data.shape)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(data, edgecolors='gray')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(' voxels')
    plt.show()
    label = input("label: ")

    with open('dataset/' + label + '/' + inst + 'f1.pkl', "wb") as f:
        pickle.dump(data, f)
        i = i + 1