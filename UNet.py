from skimage.io import imread_collection
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from unet import model
import numpy as np


# Creacion del modelo
network = model.unet(input_size=(256, 256,3))
network.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metric=['accuracy'])
print(network.summary())

# Carga de las imagenes 
path = 'floyd/input/dataset'
ims = imread_collection(path + '*.jpg')
targets = imread_collection(path + '*.png')

print('Numero de imagenes: ', len(ims))

plt.figure()
plt.subplot(121)
plt.title('Imagen')
plt.imshow(ims[0])
plt.subplot(122)
plt.title('Segmentacion')
plt.imshow(targets[0], cmap='gray')
plt.show()


# Entrenar la red neuronal
#history = network.fit(ims[:3000], targets[:3000], epochs=5, batch_size=1, shuffle=True)

# Predecir la segmentacion
#segmented = network.predict(ims[3000:])