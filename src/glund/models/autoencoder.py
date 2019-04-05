# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/snatch59/keras-autoencoders

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K

class Autoencoder:
    def __init__(self, length=28*28, dim=100):
        self.length = length
        self.dim = dim
        self.build_AE()

    def build_AE(self):
        """Construct an autoencoder"""
        inputs = Input(shape=(self.length,))
        encoded = Dense(1024, activation='relu')(inputs)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(self.dim, activation='relu')(encoded)
        
        decoded = Dense(512, activation='relu')(encoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(self.length, activation='sigmoid')(decoded)

        self.autoencoder = Model(inputs, decoded)
        self.encoder     = Model(inputs, encoded)
        self.autoencoder.summary()
        encoded_inputs = Input(shape=(self.dim, ))
        # retrieve the layers of the autoencoder model
        decoder_layer1 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer3 = self.autoencoder.layers[-1]
        outputs = decoder_layer3(decoder_layer2(decoder_layer1(encoded_inputs)))
        # create the decoder model
        self.decoder = Model(encoded_inputs, outputs)
        # configure model to use a per-pixel binary crossentropy loss,
        # and the Adadelta optimizer
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, X_train, epochs, batch_size=128, sample_interval=None):
        # autoencode the model
        self.autoencoder.fit(X_train, X_train, epochs=epochs,
                             batch_size=batch_size, shuffle=True)
        # self.autoencoder.fit(X_train, X_train, epochs=epochs,
        #                      batch_size=batch_size, shuffle=True,
        #                      validation_split=0.1

    def encode(self, image):
        return self.encoder.predict(image.reshape(image.shape[0],self.length))

    def decode(self, encoded_img):
        return self.decoder.predict(encoded_img).reshape(encoded_img.shape[0],self.length)
