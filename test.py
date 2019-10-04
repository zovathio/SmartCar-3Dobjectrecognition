import keras
import numpy as np
import os
from os import listdir
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model

# Modell betöltése
newModel = keras.models.load_model('/home/orkeny/PycharmProjects/3Dobject_class/bestModel.h5', custom_objects={"tf": tf})
# load weights into new model
plot_model(newModel, to_file='images/model.png')
#print(newModel.summary())
print("Loaded model from disk.")



mypath="/home/orkeny/PycharmProjects/3Dobject_class/test/"
finalpath="/home/orkeny/Documents/AIPro_minimal/"
objects = [f for f in listdir(mypath)]
objects.sort()

inputs = []
nums = []
cnt = 0
for obj in objects:
    print('test/'+obj)
    '''with open(mypath+obj, "rb") as f:
        data = pickle.load(f)
    '''
    f = open(mypath+obj, "rb")
    f.seek(os.SEEK_SET)
    data = np.fromfile(f, dtype=np.int32)
    f.close()

    data = data.reshape((40, 40, 40))

    data = data.astype(float)
    data = 2 * data - 1
    inputs.append(data)
    nums.append(obj[3:])



# data = data1+data2
inputs = np.asarray(inputs, dtype='float32')


# X_test = np.expand_dims(data, axis=0)
X_test = np.expand_dims(inputs, axis=4)
labels = newModel.predict(X_test)
outputLB = pickle.loads(open("classifier.label", "rb").read())
label = outputLB.inverse_transform(labels)
print(labels)
filename = "pred.txt"

# Open the file with writing permission
myfile = open(finalpath+filename, 'w')

# Write a line to the file
for l, n, p in zip(label, nums, labels):
    print(str(n) + ";" + l)
    # print(p)
    myfile.write(str(n) + ";" + l+'\n')

# Close the file
myfile.close()