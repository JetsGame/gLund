import matplotlib.pyplot as plt
from read_data import Jets 
from JetTree import JetTree, LundImage
import numpy as np
import os, argparse
from matplotlib.backends.backend_pdf import PdfPages
def plot_mnist(filename):
    r, c = 5, 5
    imgs = np.load(filename)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(sample[cnt, :,:], cmap='gray',vmin=0.0,vmax=2.0)
            axs[i,j].axis('off')
            cnt += 1
    figname=filename.split(os.extsep)[0]+'.pdf'
    fig.savefig(figname)
    plt.close()

def plot_lund(filename, figname):
    r, c = 5, 5
    imgs = np.load(filename)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    with PdfPages(figname) as pdf:
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(sample[cnt, :,:], cmap='gray',vmin=0.0,vmax=2.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        plt.imshow(np.average(imgs,axis=0))
        plt.close()
        pdf.savefig(fig)
        
def plot_lund_with_ref(filename, reference, figname):
    r, c = 5, 5
    imgs = np.load(filename)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    # now read in the pythia reference sample
    reader=Jets(reference, imgs.shape[0])
    events=reader.values() 
    imgs_ref=np.zeros((len(events), imgs.shape[1], imgs.shape[2]))
    li_gen=LundImage(npxlx = imgs.shape[1], npxly = imgs.shape[2]) 
    for i, jet in enumerate(events): 
        tree = JetTree(jet) 
        imgs_ref[i]=li_gen(tree).reshape(imgs.shape[1], imgs.shape[2])
    sample_ref = imgs_ref[np.random.choice(imgs_ref.shape[0], r*c, replace=False), :]
    with PdfPages(figname) as pdf:
        fig, axs = plt.subplots(r, c)
        plt.suptitle('generated')
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt, :,:], cmap='gray',vmin=0.0,vmax=2.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig, axs = plt.subplots(r, c)
        plt.suptitle('reference')
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs_ref[cnt, :,:], cmap='gray',vmin=0.0,vmax=2.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        plt.title('generated')
        plt.imshow(np.average(imgs,axis=0),vmin=0.0,vmax=0.7)
        plt.colorbar(orientation='vertical', label=r'$\rho$')
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        plt.title('reference')
        plt.imshow(np.average(imgs_ref,axis=0),vmin=0.0, vmax=0.7)
        plt.colorbar(orientation='vertical', label=r'$\rho$')
        plt.close()
        pdf.savefig(fig)

if __name__ == '__main__':
    # read in the arguments
    parser = argparse.ArgumentParser(description='Plot a model.')
    parser.add_argument('--data', type=str, default=None, help='Generated images')
    parser.add_argument('--reference', type=str, default=None, help='Pythia reference')
    args = parser.parse_args()

    if not args.data:
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
    elif args.reference:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund_with_ref(args.data, args.reference, figname)
    else:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund(args.data, figname)
