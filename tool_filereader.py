import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


mypath="/home/orkeny/PycharmProjects/3Dobject_class/dataset/facade"
classnames = [f for f in listdir(mypath)]
print(classnames)

labels = []
inputs = []

instances = [files for files in listdir(mypath+"/")]
for inst in instances:
    f = open(mypath+"/"+inst, "rb")
    f.seek(os.SEEK_SET)
    data = np.fromfile(f, dtype=np.int32)
    f.close()
    print(data.max())
    data = data.reshape((40,40,40))
    print(data.shape)
    ax = make_ax(True)
    ax.voxels(data, edgecolors='gray')
    plt.show()

