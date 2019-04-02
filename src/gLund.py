from keras.datasets import mnist
from read_data import Jets
from JetTree import JetTree, LundImage
from gan import GAN
from bgan import BGAN
from dcgan import DCGAN
from wgan_gp import WGANGP
from wgan import WGAN
from vae import VAE
from tools import loss_calc
from preprocess import PreprocessPCA, PreprocessZCA
import matplotlib.pyplot as plt
import numpy as np
import argparse, os, shutil, sys, datetime

# read in the arguments
parser = argparse.ArgumentParser(description='Train a generative model.')
parser.add_argument('--mnist',  action='store_true',
                    help='Train on MNIST data (for testing purposes).')
parser.add_argument('--gan',    action='store_true')
parser.add_argument('--dcgan',  action='store_true')
parser.add_argument('--wgan',   action='store_true')
parser.add_argument('--wgangp', action='store_true')
parser.add_argument('--vae',    action='store_true')
parser.add_argument('--bgan',    action='store_true')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
parser.add_argument('--batch-size', type=int, default=32, dest='batch_size')
parser.add_argument('--nev', '-n', type=int, default=-1,
                    help='Number of training events.')
parser.add_argument('--ngen', type=int, default=5000,
                    help='Number of generated events.')
parser.add_argument('--dim', type=int, default=100, dest='latdim', help='Number of latent dimensions.')
parser.add_argument('--npx', type=int, default=28, help='Number of pixels.')
parser.add_argument('--data', type=str,
                    default='../../data/valid/valid_QCD_500GeV.json.gz',
                    help='Data set on which to train.')
parser.add_argument('--pca', action='store',const=0.95, default=None,
                    nargs='?', type=float, help='Perform PCA.')
parser.add_argument('--zca', action='store_true', help='Perform ZCA.')
parser.add_argument('--output', type=str, required=True, help='Output folder.')
parser.add_argument('--force', action='store_true', help='Overwrite existing output if necessary')
args = parser.parse_args()

# check that input is valid
if not (args.gan+args.dcgan+args.wgan+args.wgangp+args.vae+args.bgan == 1):
    raise ValueError('Invalid input: choose one model at a time.')
if os.path.exists(args.output) and not args.force:
    raise Exception(f'{args.output} already exists, use "--force" to overwrite.')

# for GAN or VAE, we want to flatten the input and preprocess it
flat_input = args.gan or args.vae

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


# if requested, set up a preprocessing pipeline
if args.pca:
    preprocess = PreprocessPCA(args.pca, whiten=False)
elif args.zca:
    preprocess = PreprocessZCA(flatten=flat_input, remove_zero=flat_input)

# prepare the training data for the model training
if args.pca or args.zca:
    preprocess.fit(img_train)
    # NB: for ZCA, the zca factor is set in the process.transform call
    img_train = preprocess.transform(img_train)
elif flat_input:
    img_train = img_train.reshape(-1, args.npx*args.npx)

# now set up the model
if args.wgan:
    model = WGAN(width=args.npx, height=args.npx, latent_dim=args.latdim)
elif args.wgangp:
    model = WGANGP(width=args.npx, height=args.npx, latent_dim=args.latdim)
elif args.vae:
    model = VAE(length=(img_train.shape[1]), latent_dim=args.latdim, mse_loss=False)
elif args.dcgan:
    model = DCGAN(width=args.npx, height=args.npx, latent_dim=args.latdim)
elif args.gan:
    model = GAN(length=(img_train.shape[1]), latent_dim=args.latdim)
elif args.bgan:
    model = BGAN(length=(img_train.shape[1]), latent_dim=args.latdim)

# train on the images
model.train(img_train, epochs=args.epochs,
            batch_size = args.batch_size)

# now generate a test sample and save it
gen_sample = model.generate(args.ngen)

# retransform the generated sample to image space
if args.pca or args.zca:
    gen_sample = preprocess.inverse(gen_sample)
else:
    gen_sample = gen_sample.reshape(args.ngen, args.npx, args.npx)

# prepare the output folder
if not os.path.exists(args.output):
    os.mkdir(args.output)
elif args.force:
    shutil.rmtree(args.output)
    os.mkdir(args.output)
else:
    raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
folder = args.output.strip('/')

# for loss function, define epsilon and retransform the training sample
epsilon=0.05
# get reference sample and generated sample for tests
if args.pca or args.zca:
    img_train = preprocess.inverse(img_train)
ref_sample = img_train.reshape(img_train.shape[0],args.npx,args.npx)\
    [np.random.choice(img_train.shape[0], len(gen_sample), replace=True), :]

# write out a file with basic information on the run
with open('%s/info.txt' % folder,'w') as f:
    print('# %s' % model.description(), file=f)
    print('# created on %s with the command:' % datetime.datetime.utcnow(), file=f)
    print('# '+' '.join(sys.argv), file=f)
    print('# loss = %f' % loss_calc(gen_sample,ref_sample,epsilon), file=f)

# save the model to file
model.save(folder)

# save a generated sample to file and plot the average image
genfn = '%s/generated_images' % folder
np.save(genfn, gen_sample)
plt.imshow(np.average(gen_sample, axis=0))
plt.show()
