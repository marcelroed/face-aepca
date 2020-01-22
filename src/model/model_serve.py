from enum import Enum, auto
from pathlib import Path

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from src.model.model import AutoEncoder
from src.model.configure import tf_gpu

tf_gpu()

data_dir = Path(__file__).parents[2] / 'data'


class ModelServer:
    def __init__(self, weights_file: str, dataset: str):
        print(weights_file)
        self.images = np.load(f'{str(data_dir)}/{dataset}/processed_images.npy')[:100].astype(float) / 255.0

        self.names = np.load(f'{str(data_dir)}/{dataset}/processed_names.npy')[:100]

        self.model = AutoEncoder()
        self.model.compile(optimizer='adam',
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[])
        self.model.fit(self.images[:1], self.images[:1])
        self.model.load_weights(str(weights_file))

        self.encoded = self.model.enc(self.images)

        self.pca = PCA(n_components=70)
        self.pca.fit(self.encoded)

        self.encoded = self.pca.transform(self.encoded)

        self.current = -1
        self.pc_coeffs = None
        self.image = None

        self.random_person()

    def random_person(self):
        self.current = np.random.randint(0, self.names.shape[0])
        print(self.encoded.shape)
        self.pc_coeffs = self.encoded[np.newaxis, self.current]
        print(self.names[self.current])
        self.update_image()

    def update_image(self):
        from_pcs = self.pca.inverse_transform(self.pc_coeffs)

        self.image = self.model.dec(from_pcs)[0]

    def set_pc_coeff(self, i, value):
        self.pc_coeffs[0, i] = value





