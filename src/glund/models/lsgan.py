# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/lsgan

from __future__ import print_function, division

from glund.models.optimizer import build_optimizer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model

import matplotlib.pyplot as plt
import numpy as np
import sys, math


#======================================================================
class LSGAN():

    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape  = (self.length,)
        self.latent_dim = hps['latent_dim']

        # opt =  Adam(0.0002, 0.5)
        opt = build_optimizer(hps)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(units=hps['nn_smallest_unit'],alpha=hps['nn_alpha'])
        self.discriminator.compile(loss='mse',
            optimizer=opt,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(units=hps['nn_smallest_unit'],
                                              alpha=hps['nn_alpha'], momentum=hps['nn_momentum'])

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(z, valid)
        # (!!!) Optimize w.r.t. MSE loss instead of crossentropy
        self.combined.compile(loss='mse', optimizer=opt)

    #----------------------------------------------------------------------
    def build_generator(self, units=256, alpha=0.2, momentum=0.8):

        model = Sequential()

        model.add(Dense(units, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units*2))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units*4))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(self.length, activation='tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    #----------------------------------------------------------------------
    def build_discriminator(self, units=256, alpha=0.2):

        model = Sequential()

        model.add(Dense(units*2, input_shape=self.shape))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=alpha))
        # (!!!) No softmax
        model.add(Dense(1))
        model.summary()

        img = Input(shape=self.shape)
        validity = model(img)

        return Model(img, validity)

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

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch%10==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                       % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    #----------------------------------------------------------------------
    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev, self.latent_dim))
        return self.generator.predict(noise)
    
    #----------------------------------------------------------------------
    def load(self, folder):
        """Load GAN from input folder"""
        # load the weights from input folder
        self.generator.load_weights('%s/generator.h5'%folder)
        self.discriminator.load_weights('%s/discriminator.h5'%folder)

    #----------------------------------------------------------------------
    def save(self, folder):
        """Save the GAN weights to file."""
        self.generator.save_weights('%s/generator.h5'%folder)
        self.discriminator.save_weights('%s/discriminator.h5'%folder)
        
    #----------------------------------------------------------------------
    def description(self):
        descrip = 'LSGAN with length=%i, latent_dim=%i'\
            % (self.length, self.latent_dim)
        return descrip

