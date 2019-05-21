# This file is part of gLund by S. Carrazza and F. A. Dreyer
import matplotlib.pyplot as plt
from matplotlib import cm
from glund.read_data import Jets 
from glund.JetTree import JetTree, LundImage
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

    
xval = [0.0, 7.0]
yval = [-3.0, 7.0]

#----------------------------------------------------------------------
def plot_lund(filename, figname):
    """Plot a few samples of lund images as well as the average density."""
    r, c = 5, 5
    imgs = np.load(filename)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    with PdfPages(figname) as pdf:
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(sample[cnt, :,:].transpose(), origin='lower',
                                cmap='gray', vmin=0.0, vmax=2.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        plt.imshow(np.average(imgs,axis=0).transpose(), origin='lower')
        plt.close()
        pdf.savefig(fig)

        
#----------------------------------------------------------------------
def plot_lund_with_ref(filename, reference, figname):
    """Plot a samples of lund images and the average density along with reference data."""
    r, c = 5, 5
    imgs = np.load(filename)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]

    if reference == 'mnist':
        # if mnist data, load the images from keras
        from keras.datasets import mnist
        (imgs_ref, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        imgs_ref = imgs_ref.astype('float32') / 255
    else:
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
                axs[i,j].imshow(imgs[cnt, :,:].transpose(), origin='lower', cmap='gray',vmin=0.0,vmax=1.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig, axs = plt.subplots(r, c)
        plt.suptitle('reference')
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs_ref[cnt, :,:].transpose(), origin='lower', cmap='gray',vmin=0.0,vmax=1.0)
                axs[i,j].axis('off')
                cnt += 1
        pdf.savefig()
        plt.close()
        fig=plt.figure(figsize=(6,6))
        cbartics   = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        plt.title('generated')
        plt.imshow(np.average(imgs,axis=0).transpose(), origin='lower',vmin=0.0,vmax=0.2,
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.colorbar(orientation='vertical', label=r'$\rho$', ticks=cbartics)
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()
        fig=plt.figure(figsize=(6,6))
        plt.title('reference')
        plt.imshow(np.average(imgs_ref,axis=0).transpose(), origin='lower',vmin=0.0, vmax=0.2,
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.colorbar(orientation='vertical', label=r'$\rho$', ticks=cbartics)
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()
        fig=plt.figure(figsize=(6,6))
        plt.title('generated/reference')
        plt.imshow(np.divide(np.average(imgs,axis=0).transpose(), np.average(imgs_ref,axis=0).transpose()),
                   origin='lower', vmin=0.5, vmax=1.5, cmap=cm.seismic,
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.colorbar(orientation='vertical')
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        bins = np.arange(0, 101, 1)
        gen_act=[]
        ref_act=[]
        for i in range(len(imgs)):
            gen_act.append(np.sum(imgs[i]))
        for i in range(len(imgs_ref)):
            ref_act.append(np.sum(imgs_ref[i]))
        plt.hist(gen_act, bins=bins, density=True, color='C0', alpha=0.3, label='generated')
        plt.hist(ref_act, bins=bins, density=True, color='C1', alpha=0.3, label='reference')
        plt.title('activated pixels')
        plt.xlim((0,50))
        plt.legend()
        plt.close()
        pdf.savefig(fig)
