# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/wgan
# and from github.com/keras-team/keras-contrib/blob/master/examples

from __future__ import print_function, division

from glund.models.optimizer import build_optimizer

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

#======================================================================
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    #----------------------------------------------------------------------
    def _merge_function(self, inputs):
        # FD: should this be (hps['nn_smallest_unit']*2, 1, 1 ,1) now?
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


#======================================================================
class WGANGP():
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
        opt = build_optimizer(hps)

        # Build the generator and critic
        self.generator = self.build_generator(units=hps['nn_smallest_unit'],
                                              momentum=hps['nn_momentum'])
        self.critic = self.build_critic(units=hps['nn_smallest_unit'],
                                        alpha=hps['nn_alpha'],
                                        momentum=hps['nn_momentum'],
                                        dropout=hps['nn_dropout'])

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=opt,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=opt)


    #----------------------------------------------------------------------
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


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

        model.add(Conv2D(units, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
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

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

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
        descrip = 'WGAN-GP with width=%i, height=%i, latent_dim=%i'\
            % (self.img_rows, self.img_cols, self.latent_dim)
        return descrip
