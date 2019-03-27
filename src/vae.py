'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os, math

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

class VAE(object):

    def __init__(self, latent_dim=10, length=784, mse_loss=False):
        self.length = length
        self.latent_dim = latent_dim
        self.intermediate_dim = 512
        self.shape = (self.length,)
        # VAE model = encoder + decoder
        self.build_VAE(mse_loss)
        # set up everything
        
        
    def build_VAE(self, mse_loss):
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
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        self.encoder = encoder

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.length, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        self.decoder = decoder

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        
        # VAE loss = mse_loss or xent_loss + kl_loss
        if mse_loss:
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
        vae.compile(optimizer='adam')
        vae.summary()
        # plot_model(vae,to_file='vae_mlp.png',show_shapes=True)
        self.vae = vae

    def train(self,x_train, epochs=100, batch_size=128):
        x_train = np.reshape(x_train, [-1, self.length])
        self.vae.fit(x_train, epochs=epochs)

    # def train(self,x_train, x_test, epochs=100, batch_size=128):
    #     x_train = np.reshape(x_train, [-1, self.length])
    #     x_test  = np.reshape(x_test,  [-1, self.length])
    #     self.vae.fit(x_train, epochs=epochs, validation_data=(x_test, None))

    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev, self.latent_dim))
        return self.decoder.predict(noise)

    def plot_results(self):
        r, c = 5, 5
        gen_imgs = self.generate(r*c)
        # Rescale images 0 - 1
        npixel = int(math.sqrt(self.length))
        gen_imgs = gen_imgs.reshape(r*c, npixel, npixel,1)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
                    fig.savefig("images/vae_final.png")
                    plt.close()

if __name__ == '__main__':
    
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    vae = VAE()    
    # vae.vae.load_weights('vae_mlp_mnist.h5')
    # train the autoencoder
    vae.train(x_train, epochs=1000, batch_size=128)
    # vae.vae.save_weights('vae_mlp_mnist.h5')

    vae.plot_results()
