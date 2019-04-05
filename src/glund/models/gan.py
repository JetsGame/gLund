# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/gan

from glund.models.optimizer import build_optimizer
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
import matplotlib.pyplot as plt
import numpy as np
import math

#======================================================================
class GAN(object):

    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape  = (self.length,)
        self.latent_dim = hps['latdim']

        # optimizer
        #opt = Adam(lr=0.0002, decay=8e-9)
        opt = build_optimizer(hps)

        # allocate generator and discriminant
        self.generator = self.build_generator(units=hps['nn_smallest_unit'],
                                              alpha=hps['nn_alpha'], momentum=hps['nn_momentum'])
        self.generator.compile(loss='binary_crossentropy', optimizer=opt)
        self.discriminator = self.build_discriminator(units=hps['nn_smallest_unit'], alpha=hps['nn_alpha'])
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.adversarial_model = self.ad_model()
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=opt)

    #----------------------------------------------------------------------
    def train(self, x, epochs=10000, batch_size=32):
        """The train method"""
        for ite in range(epochs):

            # train discriminator
            random_index = np.random.randint(0, len(x) - batch_size//2)
            legit_images = x[random_index:random_index+batch_size//2].reshape(batch_size//2, self.length)
            gen_noise = np.random.normal(0, 1, (batch_size//2, self.latent_dim))
            syntetic_images = self.generator.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((batch_size//2, 1)), np.zeros((batch_size//2, 1))))

            d_loss = self.discriminator.train_on_batch(x_combined_batch, y_combined_batch)

            # train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            y_mislabled = np.ones((batch_size, 1))

            g_loss = self.adversarial_model.train_on_batch(noise, y_mislabled)
            
            if ite%10==0:
                print ("%d [D loss: %f] [G loss: %f]" % (ite, d_loss[0], g_loss))

    #----------------------------------------------------------------------
    def build_generator(self, units=256, alpha=0.2, momentum=0.8):
        """The GAN generator"""
        model = Sequential()
        model.add(Dense(units, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units*2))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units*4))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(self.length, activation='tanh'))
        return model

    #----------------------------------------------------------------------
    def build_discriminator(self, units=256, alpha=0.2):
        """The GAN discriminator"""
        model = Sequential()
        model.add(Dense(units*2, input_shape=self.shape))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    #----------------------------------------------------------------------
    def ad_model(self):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    #----------------------------------------------------------------------
    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev,self.latent_dim))
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
        return 'GAN with length=%i, latent_dim=%i' % (self.length, self.latent_dim)

