from keras.datasets import mnist
from read_data import Jets
from JetTree import JetTree, LundImage
from autoencoder import Autoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tools import zca_whiten, loss_calc
import matplotlib.pyplot as plt
import numpy as np
import argparse, os, shutil, sys, datetime

# read in the arguments
parser = argparse.ArgumentParser(description='Train a generative model.')
parser.add_argument('--mnist',  action='store_true',
                    help='Train on MNIST data (for testing purposes).')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
parser.add_argument('--batch-size', type=int, default=32, dest='batch_size')
parser.add_argument('--nev', '-n', type=int, default=-1,
                    help='Number of training events.')
parser.add_argument('--dim', type=int, default=100, dest='latdim',
                    help='Number of latent dimensions.')
parser.add_argument('--npx', type=int, default=28, help='Number of pixels.')
parser.add_argument('--data', type=str,
                    default='../../data/valid/valid_QCD_500GeV.json.gz',
                    help='Data set on which to train.')
parser.add_argument('--int', action='store_true',
                    help='Convert output values to nearest integer.')
parser.add_argument('--pca', action='store',const=0.95, default=None,
                    nargs='?', type=float, help='Perform PCA.')
parser.add_argument('--zca', action='store_true', help='Perform ZCA.')
parser.add_argument('--autoencoder', action='store',const=200, default=None,
                    nargs='?', type=int, help='Perform autoencoding')
args = parser.parse_args()


# read in the data set
if args.mnist:
    # for debugging purposes, we have the option of loading in the
    # mnist data and training the model on this.
    (img_train, _), (_, _) = mnist.load_data()
    # Rescale -1 to 1
    if not args.vae:
        img_train = (img_train.astype(np.float32) - 127.5) / 127.5
    else:
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
        
img_train = img_train.reshape(img_train.shape[0], args.npx*args.npx)
if args.pca:
    scaler = StandardScaler()
    scaler.fit(img_train)
    img_transf = scaler.transform(img_train)
    pca = PCA(args.pca, whiten=True)
    pca.fit(img_transf)
    img_transf = pca.transform(img_transf)
    print('Shape after PCA:',img_transf.shape)
elif args.autoencoder:
    scaler = StandardScaler()
    scaler.fit(img_train)
    img_transf = scaler.transform(img_train)
    ae = Autoencoder(width=args.npx, height=args.npx, dim=args.autoencoder)
    ae.train(img_transf, args.epochs)
    img_transf = ae.encode(img_transf)
    print('Shape after autoencoding:',img_transf.shape)
elif args.zca:
    scaler = StandardScaler()
    scaler.fit(img_train)
    img_transf = scaler.transform(img_train)
    img_transf, zca = zca_whiten(img_transf)
    print('Shape after ZCA:',img_transf.shape)

# now transform back to images
if args.pca:
    img_transf = scaler.inverse_transform(pca.inverse_transform(img_transf))
elif args.autoencoder:
    img_transf = scaler.inverse_transform(ae.decode(img_transf).reshape(img_transf.shape[0],args.npx*args.npx))
elif args.zca:
    img_transf = scaler.inverse_transform(np.dot(img_transf, zca))

r=5
selec=np.random.choice(img_transf.shape[0], 2*r, replace=True)
ref_transf = np.round(img_transf.reshape(img_transf.shape[0],args.npx,args.npx)[selec, :])
ref_train = img_train.reshape(img_train.shape[0],args.npx,args.npx)[selec, :]


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

plt.plot([0.5, 0.5], [0, 1], color='lightgray',
lw=5,transform=plt.gcf().transFigure, clip_on=False)
plt.show()
plt.close()
