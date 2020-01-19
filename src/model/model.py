import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from os import listdir
import re
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.1
ENCODER_TRAINABLE_LAYERS = 30
retrain = True

data_dir = '../../data/famousbirthdays/'
model_dir = './saved_models/'

DIM = np.array([64, 64, 3])


def create_imagenet_model():
    base_model = keras.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_shape=tuple(DIM))

    for layer in base_model.layers[:-ENCODER_TRAINABLE_LAYERS]:
        layer.trainable = False
        if not len(layer.get_weights()):
            continue

    for layer in base_model.layers[-ENCODER_TRAINABLE_LAYERS:]:
        new_weights = [np.full(fill_value=1.0, shape=w.shape, dtype=w.dtype) for w in layer.get_weights()]
        layer.set_weights(new_weights)

    # base_model.summary()

    lambdas = [keras.layers.Lambda(
        lambda image, i=i: tf.image.resize(image, size=(2 ** i, 2 ** i), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ) for i in range(2, 7)]  # Upscale

    conv2ds = [keras.layers.Conv2D(filters, (3, 3), padding='same', activation=keras.activations.relu) for filters in
               (220, 200, 160, 120, 15)]

    encoder_layers = [None] * (len(lambdas) + len(conv2ds))
    encoder_layers[::2] = lambdas
    encoder_layers[1::2] = conv2ds

    encoder_layers += [
        keras.layers.Conv2D(3, (3, 3), padding='same', activation=None),
        keras.layers.Activation('relu')
    ]

    output = base_model.output

    for layer in encoder_layers:
        output = layer(output)

    # Create and compile extended model
    extended_model = keras.Model(inputs=base_model.input, outputs=output)
    extended_model.compile(optimizer='adam', loss='mse')
    extended_model.summary()
    return extended_model


class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=120, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv2 = keras.layers.Conv2D(filters=160, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3 = keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv4 = keras.layers.Conv2D(filters=240, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten = keras.layers.Reshape([-1, 4 * 4 * 240])
        self.reduce_inner = keras.layers.Dense(300)
        self.activate_inner = keras.layers.Activation(keras.activations.relu)
        self.expand_inner = keras.layers.Dense(4 * 4 * 240)
        self.un_flatten = keras.layers.Reshape([-1, 4, 4, 240])
        self.up_sample = lambda s: lambda image: tf.image.resize_images(
                image,
                (s, s),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True,  # possibly important
                preserve_aspect_ratio=True
        )
        self.conv5 = keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.conv6 = keras.layers.Conv2D(filters=160, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.conv7 = keras.layers.Conv2D(filters=120, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.conv8 = keras.layers.Conv2D(filters=15, kernel_size=(3, 3), padding='same', activation=keras.activations.relu)
        self.conv9 = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation=None)
        self.decode = keras.layers.Activation(keras.activations.sigmoid)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.reduce_inner(x)
        x = self.activate_inner(x)
        x = self.expand_inner(x)
        x = self.un_flatten(x)
        x = self.up_sample(8)(x)
        x = self.conv5(x)
        x = self.up_sample(16)(x)
        x = self.conv6(x)
        x = self.up_sample(32)(x)
        x = self.conv7(x)
        x = self.up_sample(64)(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.decode(x)
        return x


# Load data
images = np.load(f'{data_dir}processed_images.npy')[:3].astype(float) / 255.0
print(images[0])

model = AutoEncoder()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])


# Get partially trained model if it exists
newest_model = keras.models.load_model(model_dir + max(listdir(model_dir))) if len(
    listdir(model_dir)) and not retrain else create_imagenet_model()


model_no = int(re.search('([0-9]+)', max(listdir(model_dir))).groups()[0]) + 1 if len(listdir(model_dir)) else 0
print(f'model_no = {model_no}')

# Train model
history = newest_model.fit(images,
                           images,
                           epochs=100,
                           callbacks=[keras.callbacks.ModelCheckpoint(
                               filepath=f'{model_dir}/model{str(model_no).zfill(3)}.h5',
                               save_best_only=True
                           )],
                           validation_split=0.10)


print(newest_model.predict(images[:2]))

# Save model
newest_model.save('model.h5')
