# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""This script provides diagnostic plots for generated lund images"""

import matplotlib.pyplot as plt
from matplotlib import cm
from glund.read_data import Jets 
from glund.JetTree import JetTree, LundImage
import numpy as np
import os, argparse
from matplotlib.backends.backend_pdf import PdfPages

    
#----------------------------------------------------------------------
def plot_lund(filename, figname, eps=None, rnd=False):
    """Plot a few samples of lund images as well as the average density."""
    r, c = 5, 5
    imgs = np.load(filename)
    if eps:
        imgs[imgs<eps]=0
    if rnd:
        imgs = np.round(imgs)
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

        
#----------------------------------------------------------------------
def plot_lund_with_ref(filename, reference, figname, eps=None,
                       rnd=False, prob=False):
    """Plot a samples of lund images and the average density along with reference data."""
    r, c = 5, 5
    imgs = np.load(filename)
    if eps:
        imgs[imgs<eps]=0.0
    if rnd:
        imgs = np.round(imgs)
    sample = imgs[np.random.choice(imgs.shape[0], r*c, replace=False), :]
    # now read in the pythia reference sample
    reader=Jets(reference, imgs.shape[0])
    events=reader.values() 
    imgs_ref=np.zeros((len(events), imgs.shape[1], imgs.shape[2]))
    li_gen=LundImage(npxlx = imgs.shape[1], npxly = imgs.shape[2],
                     norm_to_one=prob) 
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
        pdf.savefig()
        plt.close()
        fig=plt.figure()
        plt.title('generated/reference')
        plt.imshow(np.divide(np.average(imgs,axis=0),np.average(imgs_ref,axis=0)),
                   vmin=0.5, vmax=1.5, cmap=cm.seismic)
        plt.colorbar(orientation='vertical')
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
        plt.hist(gen_act, bins=bins, color='C0', alpha=0.3, label='generated')
        plt.hist(ref_act, bins=bins, color='C1', alpha=0.3, label='reference')
        plt.title('activated pixels')
        plt.xlim((0,50))
        plt.legend()
        plt.close()
        pdf.savefig(fig)

        
#----------------------------------------------------------------------
def main():
    # read in the arguments
    parser = argparse.ArgumentParser(description='Plot a model.')
    parser.add_argument('--data', type=str, required=True, help='Generated images')
    parser.add_argument('--reference', type=str, default=None, help='Pythia reference')
    parser.add_argument('--round', action='store_true',
                        help='Round all pixel values to nearest integer')
    parser.add_argument('--epsilon', action='store', default=None,
                        type=float, help='Threshold for pixel activation.')
    parser.add_argument('--deterministic', action='store_true', dest='det',
                        help='Round all pixel values to nearest integer')
    args = parser.parse_args()

    if args.reference:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund_with_ref(args.data, args.reference, figname,
                           args.epsilon, args.round, not args.det)
    else:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund(args.data, figname, args.epsilon, args.round)
