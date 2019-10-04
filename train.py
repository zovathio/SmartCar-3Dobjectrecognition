######### IMPORT #########
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers import Dense, Dropout,InputLayer, BatchNormalization
from keras.layers import Conv3D, GlobalAveragePooling3D
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
import pickle
import keras.backend as K

def global_average_pooling(x):
    return K.mean(x, axis=(1, 2, 3))


def global_average_pooling_shape(input_shape):
    idx = []
    return input_shape[0:1:2]





mypath="/home/orkeny/PycharmProjects/3Dobject_class/dataset/"
classnames = [f for f in listdir(mypath)]
classnames.sort()
print(classnames)

labels = []
inputs = []
pathes = []
for myclass in classnames:
    instances = [files for files in listdir(mypath+myclass+"/")]
    for inst in instances:
        fpath = mypath + myclass + "/" + inst
        pathes.append(fpath)
        labels.append(myclass)
# Shuffle
d = {'x': pathes, 'y': labels}
df = pd.DataFrame(data=d)
print(df.head())
from sklearn.utils import shuffle
df = shuffle(df, random_state=177)
print(df.head())
p_inputs = df['x'].values
p_labels = df['y'].values

for path_ in p_inputs:
    with open(path_, "rb") as f:
        data = pickle.load(f)
    data = data.reshape((40, 40, 40))
    if data.min() != 0 or data.max() != 1:
        print(data.min())
        print(data.max())
    inputs.append(data)

X_train = 2 * np.asarray(inputs, dtype='float32') - 1
print(X_train.min())
print(X_train.max())

X_train = np.expand_dims(X_train, axis=4)

Y_train = np.asarray(p_labels)
lb_classes = LabelBinarizer()
Y_train = lb_classes.fit_transform(Y_train)


print(X_train.shape)
print(Y_train.shape)


# SEED
seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# Paraméterek
batch_size = 32
nb_classes = 4

dim_x, dim_y, dim_z = 40, 40, 40

nb_filters1 = 8
nb_filters2 = 16
nb_filters3 = 32
nb_filters4 = 64


nb_pool = 2
nb_pool2 = 2
nb_pool3 = 2

# Convolution kernel size
nb_conv = 3
nb_conv2 = 5
nb_conv3 = 7

nb_epoch = 100

def lr_scheduler(lr):
    return lr /(lr +1.0)

def build_model(nb_epoch, nb_classes):

    model = Sequential()
    model.add(InputLayer(input_shape=(dim_x, dim_y, dim_z,1)))

    model.add(Conv3D(nb_filters1,(nb_conv, nb_conv, nb_conv), activation='relu',name='Conv3_1'))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())

    model.add(Conv3D(nb_filters2, (nb_conv, nb_conv, nb_conv), name='Conv3_2', activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())

    model.add(Conv3D(nb_filters3, (nb_conv2, nb_conv2, nb_conv2), name='Conv3_3', activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())

    model.add(Conv3D(nb_filters3, (nb_conv, nb_conv, nb_conv), dilation_rate=[1, 1, 1], name='Conv3_4', activation='relu'))
    model.add(Dropout(rate=0.3))

    model.add(GlobalAveragePooling3D())

    model.add(Dense(nb_classes, activation='softmax'))

    learning_rate = 0.0005
    decay_rate = learning_rate / nb_epoch # L2 regularizáció
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    print('Model is compiled ...')


    return model

modelNet = build_model(nb_epoch=nb_epoch, nb_classes=nb_classes)
print(modelNet.summary())

filepath = "bestWeights.hdf5"
callbacks = [
    CSVLogger('log.csv', append=True, separator=';'),
    ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1),
    LearningRateScheduler(lr_scheduler)
]
class_weight = {0: 1.2,
                1: 1.5,
                2: 2.0,
                3: 1.0}
hist = modelNet.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks, class_weight=class_weight)

# Legjobb súlyok visszatöltése

modelNet.load_weights('bestWeights.hdf5')
print(modelNet.summary())

modelNet.save("bestModel.h5")
print("Saved model to disk.")

# Binarizáló kimentése file-ba
print("[INFO] serializing category label binarizer...")
f = open("classifier.label", "wb")
f.write(pickle.dumps(lb_classes))
f.close()