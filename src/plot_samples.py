import matplotlib.pyplot as plt
import numpy as np
import os
def plot_sample(filename):
    r, c = 5, 5
    imgs = np.load(filename)
    imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs[cnt, :,:], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    figname=filename.split(os.extsep)[0]+'.pdf'
    fig.savefig(figname)
    plt.close()


plot_sample('test/mnist_dcgan.npy')
plot_sample('test/mnist_gan.npy')
plot_sample('test/mnist_vae.npy')
plot_sample('test/mnist_wgan.npy')
plot_sample('test/mnist_wgangp.npy')
