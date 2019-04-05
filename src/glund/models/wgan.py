# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/wgan

from __future__ import print_function, division

from glund.models.optimizer import build_optimizer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import sys



#======================================================================
class WGAN():
    #----------------------------------------------------------------------
    def __init__(self, hps):
        if (hps['npx']%4):
            raise ValueError('WGAN: Width and height need to be divisible by 4.')
        self.img_rows = hps['npx']
        self.img_cols = hps['npx']
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.latent_dim = hps['latdim']

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = hps['n_critic']
        self.clip_value = hps['clip_value']
        opt = optimizer(hps)

        # Build and compile the critic
        self.critic = self.build_critic(units=hps['nn_smallest_unit'],
                                        alpha=hps['nn_alpha'],
                                        momentum=hps['nn_momentum'],
                                        dropout=hps['nn_dropout'])
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=opt,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(units=hps['nn_smallest_unit'],
                                              momentum=hps['nn_momentum'])

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=opt,
                              metrics=['accuracy'])

    #----------------------------------------------------------------------
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    #----------------------------------------------------------------------
    def build_generator(self, units=16, momentum=0.8):

        model = Sequential()

        model.add(Dense(units * self.img_rows * self.img_cols//2,
                        activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, units*8)))
        model.add(UpSampling2D())
        model.add(Conv2D(units*8, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(units*4, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(Conv2D(1, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    #----------------------------------------------------------------------
    def build_critic(self, units=16, alpha=0.2, momentum=0.8, dropout=0.25):

        model = Sequential()

        model.add(Conv2D(units, kernel_size=3, strides=2,
                         input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*2, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*4, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Conv2D(units*8, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    #----------------------------------------------------------------------
    def train(self, X_train, epochs, batch_size=128):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

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

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

    #----------------------------------------------------------------------
    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev, self.latent_dim))
        return self.generator.predict(noise)
    
    #----------------------------------------------------------------------
    def load(self, folder):
        """Load GAN from input folder"""
        # load the weights from input folder
        self.generator.load_weights('%s/generator.h5'%folder)
        self.critic.load_weights('%s/critic.h5'%folder)

    #----------------------------------------------------------------------
    def save(self, folder):
        """Save the GAN weights to file."""
        self.generator.save_weights('%s/generator.h5'%folder)
        self.critic.save_weights('%s/critic.h5'%folder)

    #----------------------------------------------------------------------
    def description(self):
        descrip = 'WGAN with width=%i, height=%i, latent_dim=%i'\
            % (self.img_rows, self.img_cols, self.latent_dim)
        return descrip
