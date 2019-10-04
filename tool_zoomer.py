import numpy as np
from scipy.ndimage.interpolation import zoom

x=np.ndarray((38, 38, 38))

y = zoom(x, 0.5)
x_augmented = []

for scale in [0.5]:
    y_sc = zoom(x, scale)
    y_pd = np.pad(y_sc,((10,9),(10,9),(10,9)),mode= 'constant')
    print(y_pd.shape)


