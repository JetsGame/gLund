# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/blob/master/aae

from glund.models.optimizer import build_optimizer

from keras.layers import Input, Dense
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import keras.backend as K

import numpy as np

#======================================================================
class AdversarialAutoencoder():
    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape = (self.length,)
        self.latent_dim = hps['latdim']

        #opt = Adam(0.0002, 0.5)
        opt = build_optimizer(hps)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(units=hps['nn_smallest_unit'],
                                                      alpha=hps['nn_alpha'])
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder(units=hps['nn_smallest_unit'],
                                          alpha=hps['nn_alpha'])
        self.decoder = self.build_decoder(units=hps['nn_smallest_unit'],
                                          alpha=hps['nn_alpha'])

        img = Input(shape=self.shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
                                             loss_weights=[0.999, 0.001],
                                             optimizer=opt)


    #----------------------------------------------------------------------
    def build_encoder(self, units=256, alpha=0.2):
        # Encoder

        img = Input(shape=self.shape)

        h = Dense(units*2)(img)
        h = LeakyReLU(alpha=alpha)(h)
        h = Dense(units*2)(h)
        h = LeakyReLU(alpha=alpha)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)

        latent_repr = Lambda(
            lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
            output_shape=lambda p: p[0]
        )([mu, log_var])
        # latent_repr = merge([mu, log_var],
        #                     mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
        #                     output_shape=lambda p: p[0])

        return Model(img, latent_repr)

    #----------------------------------------------------------------------
    def build_decoder(self, units=256, alpha=0.2):

        model = Sequential()

        model.add(Dense(units*2, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units*2))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(self.length, activation='tanh'))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    #----------------------------------------------------------------------
    def build_discriminator(self, units=256, alpha=0.2):

        model = Sequential()

        model.add(Dense(units*2, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    #----------------------------------------------------------------------
    def train(self, X_train, epochs, batch_size=128):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]"
                   % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    #----------------------------------------------------------------------
    def generate(self, nev):
        z = np.random.normal(size=(nev, self.latent_dim))
        return self.decoder.predict(z)

    #----------------------------------------------------------------------
    def load(self, folder):
        """Load GAN from input folder"""
        # load the weights from input folder
        self.encoder.load_weights('%s/encoder.h5'%folder)
        self.decoder.load_weights('%s/decoder.h5'%folder)
        self.discriminator.load_weights('%s/discriminator.h5'%folder)

    #----------------------------------------------------------------------
    def save(self, folder):
        self.encoder.save_weights('%s/encoder.h5'%folder)
        self.decoder.save_weights('%s/decoder.h5'%folder)
        self.discriminator.save_weights('%s/discriminator.h5'%folder)
        
    #----------------------------------------------------------------------
    def description(self):
        descrip = 'AAE with length=%i, latent_dim=%i'\
            % (self.length, self.latent_dim)
        return descrip

