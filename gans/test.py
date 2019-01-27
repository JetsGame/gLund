#!/usr/bin/env python
import numpy as np
from keras.datasets import mnist
from gan import GAN


if __name__ == '__main__':
    print('testing...')
    (X_train, _), (_, _) = mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    gan = GAN()
    gan.train(X_train)