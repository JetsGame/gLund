import numpy as np
from keras import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class GAN(object):

    def __init__(self, width=28, height=28, channels=1):
        self.width = width
        self.heigh = height
        self.channels = channels
        self.shape = (self.width, self.heigh, self.channels)

        # optimizer
        self.optimizer = Adam(lr=0.0002, decay=8e-9)

        # noise
        self.noise = np.random.normal(0, 1, (100,))

        # allocate generator and discriminant
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.adversarial_model = self.ad_model()
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def train(self, x, epochs=10000, batch=32, save_interval=500):
        """The train method"""
        for ite in range(epochs):

            # train discriminator
            random_index = np.random.randint(0, len(x) - batch//2)
            legit_images = x[random_index:random_index+batch//2].reshape(batch//2, self.width, self.heigh, self.channels)
            gen_noise = np.random.normal(0, 1, (batch//2, 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((batch//2, 1)), np.zeros((batch//2, 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            # train generator
            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.adversarial_model.train_on_batch(noise, y_mislabled)
            
            if ite % save_interval == 0 : 
                print (f'epoch: {ite}, Discriminator = d_loss: {d_loss[0]}, Generator = g_loss: {g_loss}]')
                self.plot_images(step=ite)

    def generator(self):
        """The GAN generator"""
        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width*self.heigh*self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.heigh, self.channels)))
        return model

    def discriminator(self):
        """The GAN discriminator"""
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width*self.heigh*self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense((self.width*self.heigh*self.channels)//2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def ad_model(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model

    def plot_images(self, samples=16, step=0):
        filename = f"./mnist_{step}.png"
        noise = np.random.normal(0, 1, (samples,100))

        images = self.G.predict(noise)
        
        plt.figure(figsize=(10,10))
    
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.heigh, self.width])
            plt.imshow(image, cmap='binary')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
