# This file is part of gLund by S. Carrazza and F. A. Dreyer
# adapted from: github.com/eriklindernoren/Keras-GAN/tree/master/lsgan

from glund.models.optimizer import build_optimizer

from keras.layers import Input, Dense, Lambda, merge, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations
import keras.backend as K

import numpy as np

#======================================================================
# MinibatchDiscrimination
# adapted from: https://github.com/keras-team/keras/pull/3677
class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#======================================================================
class LSGAN():

    #----------------------------------------------------------------------
    def __init__(self, hps, length=28*28):
        self.length = length
        self.shape  = (self.length,)
        self.latent_dim = hps['latdim']

        # opt =  Adam(0.0002, 0.5)
        opt = build_optimizer(hps)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(units=hps['nn_units_d'],
                                                      alpha=hps['nn_alpha_d'],
                                                      minibatch_discriminator=\
                                                      hps['nn_minibatch_discriminator'],
                                                      nb_kernels=hps['nn_nb_kernels'],
                                                      kernel_dim=hps['nn_kernel_dim'],
                                                      aux_ratio=hps['nn_aux_ratio'])

        self.discriminator.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        # Build the generator
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
    def build_discriminator(self, units=256, alpha=0.2, minibatch_discriminator=True,
                            nb_kernels=20, kernel_dim=10, aux_ratio=5):

        model = Sequential()

        model.add(Dense(units*2, input_shape=self.shape))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=alpha))
        if minibatch_discriminator:
            model.add(Dense(nb_kernels*aux_ratio))
            model.add(MinibatchDiscrimination(nb_kernels,kernel_dim))
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

