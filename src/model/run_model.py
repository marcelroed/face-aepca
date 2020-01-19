import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
from scipy.stats import describe

model_dir = './saved_models/'
data_dir = '../../data/famousbirthdays/'

# model = keras.models.load_model(model_dir + 'model001.h5')
model = keras.models.load_model('model.h5')

images = np.load(f'{data_dir}processed_images.npy')[:100].astype(float) / 255

out_images = model.predict(images)

plt.imshow(out_images[0])
plt.show()

# print(out_images.shape)

for image, predicted in zip(images, out_images):
    scaled_image = cv2.resize(image*255.0, (0, 0), fx=4, fy=4).astype(np.uint8)
    # print(scaled_image)
    # print(scaled_image.shape)
    scaled_prediction = cv2.resize(predicted*255.0, (0, 0), fx=4, fy=4).astype(np.uint8)
    print(predicted)
    cv2.imshow('Image', scaled_image)
    cv2.imshow('Predicted', scaled_prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
