import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
def plot_mnist(filename):
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

def plot_lund(filename):
    r, c = 5, 5
    imgs = np.load(filename)
    imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    figname=filename.split(os.extsep)[0]+'.pdf'
    with PdfPages(figname) as pdf:
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt, :,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        imgs = np.load(filename)
        print(imgs.shape)
        plt.imshow(np.average(imgs,axis=0))
        plt.close()
        pdf.savefig(fig)

plot_mnist('test/mnist_dcgan.npy')
plot_mnist('test/mnist_gan.npy')
plot_mnist('test/mnist_vae.npy')
plot_mnist('test/mnist_wgan.npy')
plot_mnist('test/mnist_wgangp.npy')

plot_lund('test/lund_dcgan.npy')
plot_lund('test/lund_gan.npy')
plot_lund('test/lund_vae.npy')
plot_lund('test/lund_wgan.npy')
plot_lund('test/lund_wgangp.npy')
