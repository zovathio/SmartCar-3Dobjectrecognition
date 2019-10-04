from keras.models import *
from keras.callbacks import *
from keras.models import load_model
import keras.backend as K
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom


def make_ax(fig,i=1,grid=False):
    ax = fig.add_subplot(1,2,i,projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

# Szintetizált összeragadt adat
inputs = []
with open('dataset/pedestrian/y_szt_909.pkl', "rb") as f:
    data1 = pickle.load(f)
data1 = data1.reshape((40, 40, 40))
with open('dataset/column/x_1593.pkl', "rb") as f:
    data2 = pickle.load(f)
data2 = data2.reshape((40, 40, 40))

data = np.copy(data1+data2)
inputs.append(data)
data_orig = np.copy(data)
data = data.astype(float)
data = 2*data-1

print(data.shape)

# Formázás a bemenetre
X_test = np.expand_dims(data, axis=3)
X_test = np.expand_dims(X_test, axis=0)

print(X_test.shape)
K.set_learning_phase(0)

# Modell betöltése
hModel = load_model('/home/orkeny/PycharmProjects/3Dobject_class/bestModel.h5', custom_objects={"tf": tf})
print(hModel.summary())


last_layer_weights = hModel.layers[-1].get_weights()[0]
print(hModel.layers[-1].name, hModel.layers[-1].get_weights()[0].shape)

# Kimenet meghatározása
# outputLB = pickle.loads(open("classifier.label", "rb").read())
outputSTR = ["car", "column","facade", "pedestrian"]

preds = hModel.predict(X_test)
print(preds[0])

class_idx = np.argmax(preds[0])
print(hModel.output)
print(class_idx)
idx = class_idx
class_output = hModel.output[:, idx]
print(class_output)
w = last_layer_weights[:, idx]
last_conv_layer = hModel.get_layer("Conv3_4")
print(last_conv_layer.output[0])

# last_pool_layer = hModel.layers[-3]
# grads = K.gradients(class_output, last_conv_layer.output)[0]
# pooled_grads = K.gradients(class_output, last_pool_layer.output)[0]
# print(pooled_grads.shape)

iterate = K.function([hModel.input], [last_conv_layer.output[0]])

f_last_conv = iterate([X_test])[0]
print(f_last_conv.shape)
print(w.shape)
sum_weights = np.zeros(f_last_conv.shape)
l_size = 30
heatmap = np.zeros((l_size,l_size,l_size), dtype='float32')
for i in range(30):
    sum_weights[:, :, :, i] += f_last_conv[:, :, :, i]*w[i]
    heatmap[:, :, :] += sum_weights[:, :, :, i]

#heatmap = np.mean(sum_weights, axis=3)
print(heatmap.shape)
# heatmap = zoom(heatmap, (2, 2, 2))
# print(heatmap.shape)
heatmap = zoom(heatmap, (40/l_size, 40/l_size, 40/l_size))
print(np.max(heatmap))
print(np.min(heatmap))

# ReLu a kimeneten
heatmap = np.maximum(heatmap,0)

if np.max(heatmap) != 0:
    heatmap/=np.max(heatmap)
    print(np.min(heatmap))
    print(np.max(heatmap))
    # label = outputLB.inverse_transform(preds)
    label = outputSTR[idx]
    print(str(label))
else:
    print('Not on the pics!')
#heatmap = np.float64(heatmap)


def k_largest_index(a, k):
    idx = np.argpartition(-a.ravel(),k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))

# Thesholdolás
n = 500
idx = k_largest_index(heatmap,n)
img = np.zeros(heatmap.shape)
for x,y,z in idx:
    #print(i.shape)
    img[x,y,z] = heatmap[x,y,z]

# img = zoom(img, (40/32, 40/32, 40/32))
img = np.maximum(img, 0) #ReLu

colors = np.zeros(data_orig.shape + (4,))
colors[..., 0] = img
colors[..., 1] = np.zeros(data_orig.shape)
colors[..., 2] = np.zeros(data_orig.shape)
colors[..., 3] = np.ones(data_orig.shape,dtype=float)


fig = plt.figure()
ax = make_ax(fig,1,True)
ax.set_title("Activization map of class: "+str(label))
ax.voxels(img,edgecolors='gray',facecolors=colors)
ax2 = make_ax(fig,2,True)
ax2.set_title("Original input voxels")
ax2.voxels(data_orig,edgecolors='gray')
plt.show()
