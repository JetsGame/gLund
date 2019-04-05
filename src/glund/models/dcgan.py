# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/dcgan

from __future__ import print_function, division

from glund.models.optimizer import build_optimizer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

import matplotlib.pyplot as plt

import sys

import numpy as np

#======================================================================
class DCGAN():
    #----------------------------------------------------------------------
    def __init__(self, hps):
        if (hps['npx']%4):
            raise ValueError('WGAN: Width and height need to be divisible by 4.')
        # Input shape
        self.img_rows = hps['npx']
        self.img_cols = hps['npx']
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.latent_dim = hps['latdim']

        #opt = Adam(0.0002, 0.5)
        opt = build_optimizer(hps)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(units=hps['nn_smallest_unit'],
                                                      alpha=hps['nn_alpha'],
                                                      momentum=hps['nn_momentum'],
                                                      dropout=hps['nn_dropout'])
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(units=hps['nn_smallest_unit'], momentum=hps['nn_momentum'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt)

    #----------------------------------------------------------------------
    def build_generator(self, units=32, momentum=0.8):

        model = Sequential()

        model.add(Dense(units * self.img_rows * self.img_cols//4,
                        activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, units*4)))
        model.add(UpSampling2D())
        model.add(Conv2D(units*4, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(units*2, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    #----------------------------------------------------------------------
    def build_discriminator(self, units=32, alpha=0.2, momentum=0.8, dropout=0.25):

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
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    #----------------------------------------------------------------------
    def train(self, X_train, epochs, batch_size=128, save_interval=None):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch%5==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                       % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if save_interval and epoch % save_interval == 0:
                self.save_imgs(epoch)

    #----------------------------------------------------------------------
    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        return gen_imgs

    #----------------------------------------------------------------------
    def save_imgs(self, epoch):
        r, c = 5, 5
        gen_imgs = self.generate(r*c)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/dcgan_%d.png" % epoch)
        plt.close()
    
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
        descrip = 'DCGAN with width=%i, height=%i, latent_dim=%i'\
            % (self.img_rows, self.img_cols, self.latent_dim)
        return descrip


#----------------------------------------------------------------------
if __name__ == '__main__':    
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    dcgan = DCGAN()
    dcgan.train(X_train, epochs=4000, batch_size=32, save_interval=50)
