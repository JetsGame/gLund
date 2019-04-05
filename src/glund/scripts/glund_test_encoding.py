# This file is part of gLund by S. Carrazza and F. A. Dreyer

from keras.datasets import mnist
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.preprocess import *
import matplotlib.pyplot as plt
import numpy as np
import argparse, os, shutil, sys, datetime

# read in the arguments
def main():
    parser = argparse.ArgumentParser(description='Train a generative model.')
    parser.add_argument('--mnist',  action='store_true',
                        help='Train on MNIST data (for testing purposes).')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--nev', '-n', type=int, default=-1,
                        help='Number of training events.')
    parser.add_argument('--dim', type=int, default=100, dest='latdim',
                        help='Number of latent dimensions.')
    parser.add_argument('--npx', type=int, default=28, help='Number of pixels.')
    parser.add_argument('--data', type=str,
                        default='../../data/valid/valid_QCD_500GeV.json.gz',
                        help='Data set on which to train.')
    parser.add_argument('--keep-zeros', action='store_true', dest='keepzero',
                        help='Convert output values to nearest integer.')
    parser.add_argument('--no-scaler', action='store_true', dest='noscaler',
                        help='Convert output values to nearest integer.')
    parser.add_argument('--no-flat', action='store_true', dest='noflat',
                        help='Convert output values to nearest integer.')
    parser.add_argument('--pca', action='store',const=0.95, default=None,
                        nargs='?', type=float, help='Perform PCA.')
    parser.add_argument('--zca', action='store_true', help='Perform ZCA.')
    parser.add_argument('--autoencoder', action='store',const=200, default=None,
                        nargs='?', type=int, help='Perform autoencoding')
    args = parser.parse_args()

    rem0=not args.keepzero
    scaler=not args.noscaler
    flatten=not args.noflat

    # read in the data set
    if args.mnist:
        # for debugging purposes, we have the option of loading in the
        # mnist data and training the model on this.
        (img_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        img_train = img_train.astype('float32') / 255
        img_train = np.expand_dims(img_train, axis=3)
    else:
        # load in the jets from file, and create an array of lund images
        reader=Jets(args.data, args.nev)
        events=reader.values() 
        img_train=np.zeros((len(events), args.npx, args.npx, 1))
        li_gen=LundImage(npxlx = args.npx) 
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
            img_train[i]=li_gen(tree).reshape(args.npx, args.npx, 1)

    # set up the preprocessing pipeline
    if args.pca:
        preprocess = PreprocessPCA(args.pca, whiten=True, scaler=scaler, flatten=flatten, remove_zero=rem0)
    elif args.autoencoder:
        preprocess = PreprocessAE(args.autoencoder, args.epochs, scaler=scaler, flatten=flatten, remove_zero=rem0)
    elif args.zca:
        preprocess = PreprocessZCA(scaler=scaler, flatten=flatten, remove_zero=rem0)
    # fit the preprocessing unit
    preprocess.fit(img_train)
    # transform the images
    # NB: for ZCA, the zca factor is set during this step
    img_transf = preprocess.transform(img_train)
    print('Shape after preprocessing:',img_transf.shape)

    # now transform back to images
    img_transf = preprocess.inverse(img_transf)

    r=5
    selec=np.random.choice(img_transf.shape[0], 2*r, replace=True)
    if args.mnist:
        ref_transf = img_transf.reshape(img_transf.shape[0],args.npx,args.npx)[selec, :]
    else:
        ref_transf = np.round(img_transf.reshape(img_transf.shape[0],args.npx,args.npx)[selec, :])
    ref_train = img_train.reshape(img_train.shape[0],args.npx,args.npx)[selec, :]

    loss=np.linalg.norm(np.average(ref_train - ref_transf,axis=0))
    print('# loss: ',loss)

    fig, axs = plt.subplots(r, 4)
    axs[0,0].title.set_text('Input')
    axs[0,3].title.set_text('Decoded')
    for i in range(r):
        axs[i,0].imshow(ref_train[i, :,:], cmap='gray')
        axs[i,0].axis('off')
        axs[i,1].imshow(ref_train[r+i, :,:], cmap='gray')
        axs[i,1].axis('off')
        axs[i,2].imshow(ref_transf[i, :,:], cmap='gray')
        axs[i,2].axis('off')
        axs[i,3].imshow(ref_transf[5+i, :,:], cmap='gray')
        axs[i,3].axis('off')

    plt.plot([0.5, 0.5], [0, 1], color='lightgray', lw=5,
            transform=plt.gcf().transFigure, clip_on=False)
    plt.show()
    plt.close()
