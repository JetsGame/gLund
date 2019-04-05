# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/keras-team/keras/blob/master/examples

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glund.models.optimizer import build_optimizer

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import math

#----------------------------------------------------------------------
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


#======================================================================
class VAE(object):

    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.latent_dim = hps['latdim']
        self.intermediate_dim = hps['nn_interm_dim']
        self.shape = (self.length,)
        self.mse_loss = hps['mse_loss']
        # VAE model = encoder + decoder
        self.build_VAE(hps)
        # set up everything
        
        
    #----------------------------------------------------------------------
    def build_VAE(self, hps):
        # build encoder model
        inputs = Input(shape=self.shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,),
                   name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        self.encoder = encoder

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.length, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        self.decoder = decoder

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        
        # VAE loss = mse_loss or xent_loss + kl_loss
        if self.mse_loss:
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs,
                                                      outputs)
        reconstruction_loss *= self.length
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        #opt = Adam(0.0003, 0.9)
        opt = build_optimizer(hps)
        vae.compile(optimizer=opt)
        # vae.compile(optimizer='adam')
        vae.summary()
        self.vae = vae

    #----------------------------------------------------------------------
    def train(self,x_train, epochs=100, batch_size=128):
        x_train = np.reshape(x_train, [-1, self.length])
        self.vae.fit(x_train, epochs=epochs)

    # def train(self,x_train, x_test, epochs=100, batch_size=128):
    #     x_train = np.reshape(x_train, [-1, self.length])
    #     x_test  = np.reshape(x_test,  [-1, self.length])
    #     self.vae.fit(x_train, epochs=epochs, validation_data=(x_test, None))

    #----------------------------------------------------------------------
    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev, self.latent_dim))
        return self.decoder.predict(noise)

    #----------------------------------------------------------------------
    def load(self, folder):
        """Load GAN from input folder"""
        # load the weights from input folder
        self.encoder.load_weights('%s/encoder.h5'%folder)
        self.decoder.load_weights('%s/decoder.h5'%folder)

    #----------------------------------------------------------------------
    def save(self, folder):
        """Save the GAN weights to file."""
        self.encoder.save_weights('%s/encoder.h5'%folder)
        self.decoder.save_weights('%s/decoder.h5'%folder)

    #----------------------------------------------------------------------
    def description(self):
        descrip = 'VAE with length=%i, latent_dim=%i, mse_loss=%s'\
            % (self.length, self.latent_dim, self.mse_loss)
        return descrip
    
