# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/bgan
# and from wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan

from glund.models.optimizer import build_optimizer

from keras.layers import Input, Dense
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import keras.backend as K

import numpy as np

#======================================================================
class BGAN():
    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape = (self.length, )
        self.latent_dim = hps['latdim']

        #optimizer = Adam(0.0002, 0.5)
        opt = build_optimizer(hps)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(units=hps['nn_units_d'],
                                                      alpha=hps['nn_alpha_d'])
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'])

        # Build the generato
        self.generator = self.build_generator(units=hps['nn_units_g'],
                                              alpha=hps['nn_alpha_g'],
                                              momentum=hps['nn_momentum'])

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.boundary_loss, optimizer=opt)

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
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.shape)
        validity = model(img)

        return Model(img, validity)

    #----------------------------------------------------------------------
    def boundary_loss(self, y_true, y_pred):
        """
        Boundary seeking loss.
        Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
        """
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)

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
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

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
        descrip = 'BGAN with length=%i, latent_dim=%i'\
            % (self.length, self.latent_dim)
        return descrip

