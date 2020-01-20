import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
from src.model.model import AutoEncoder
from sklearn.decomposition import PCA
from src.model.configure import tf_gpu

model_dir = './saved_models/'
data_dir = '../../data/famousbirthdays/'

tf_gpu()

images = np.load(f'{data_dir}processed_images.npy')[:100].astype(float) / 255.0

model = AutoEncoder()
model.compile(optimizer='adam',
              loss=keras.losses.MeanSquaredError(),
              metrics=[])
model.fit(images[:1], images[:1])
model.load_weights(model_dir + 'model004.h5')

encoded = model.enc(images)
print(encoded.shape)

# PCA model
pca = PCA(n_components=100)
pca.fit(encoded)
simplified = pca.transform(encoded)


out_images = model.dec(pca.inverse_transform(simplified))

plt.imshow(out_images[0])
plt.show()

# print(out_images.shape)

for image, predicted in zip(images, out_images):
    scaled_image = cv2.resize(image*255.0, (0, 0), fx=4, fy=4).astype(np.uint8)
    # print(scaled_image)
    # print(scaled_image.shape)
    print(type(predicted))
    scaled_prediction = cv2.resize(predicted.numpy()*255.0, (0, 0), fx=4, fy=4).astype(np.uint8)

    cv2.imshow('Image', scaled_image)
    cv2.imshow('Predicted', scaled_prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
