import numpy as np
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt


class GAN(object):

    def __init__(self, width=28, height=28):
        self.width = width
        self.heigh = height
        self.shape = (self.width, self.heigh, 1)

        # optimizer
        self.optimizer = Adam(lr=0.0002, decay=8e-9)
        #self.optimizer = RMSprop()
        
        # noise
        self.noise = np.random.normal(0, 1, (100,))

        # allocate generator and discriminant
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.adversarial_model = self.ad_model()
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def train(self, x, epochs=10000, batch_size=32, save_interval=None):
        """The train method"""
        for ite in range(epochs):

            # train discriminator
            random_index = np.random.randint(0, len(x) - batch_size//2)
            legit_images = x[random_index:random_index+batch_size//2].reshape(batch_size//2, self.width, self.heigh, 1)
            gen_noise = np.random.normal(0, 1, (batch_size//2, 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((batch_size//2, 1)), np.zeros((batch_size//2, 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            # train generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            y_mislabled = np.ones((batch_size, 1))

            g_loss = self.adversarial_model.train_on_batch(noise, y_mislabled)
            
            if save_interval and ite % save_interval == 0 : 
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
        model.add(Dense(self.width*self.heigh, activation='tanh'))
        model.add(Reshape((self.width, self.heigh, 1)))
        return model

    def discriminator(self):
        """The GAN discriminator"""
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width*self.heigh), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense((self.width*self.heigh)//2))
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

    def generate(self, nev):
        noise = np.random.normal(0, 1, (nev,100))
        return self.G.predict(noise)

    def plot_images(self, samples=16, step=0):
        filename = f"images/dcgan_{step}.png"
        images = self.generate(samples)
        
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

if __name__ == '__main__':
    
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    gan = GAN()
    gan.train(X_train, epochs=4000, batch_size=32, save_interval=50)
