from keras.datasets import mnist
from read_data import Reader, Jets 
from JetTree import *
import matplotlib.pyplot as plt
from gan import GAN
from dcgan import DCGAN
from wgan_gp import WGANGP
from wgan import WGAN
from vae import VAE
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tools import zca_whiten

parser = argparse.ArgumentParser(description='Train a generative model.')
parser.add_argument('--mnist',  action='store_true',
                    help='Train on MNIST data (for testing purposes).')
parser.add_argument('--gan',    action='store_true')
parser.add_argument('--dcgan',  action='store_true')
parser.add_argument('--wgan',   action='store_true')
parser.add_argument('--wgangp', action='store_true')
parser.add_argument('--vae',    action='store_true')
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
parser.add_argument('--pca', action='store_true', help='Perform PCA.')
parser.add_argument('--zca', action='store_true', help='Perform ZCA.')
parser.add_argument('--output', type=str, required=True, help='Output file.')
args = parser.parse_args()
# check that input is valid
if not (args.gan+args.dcgan+args.wgan+args.wgangp+args.vae == 1):
    raise ValueError('Invalid input: choose one model at a time.')
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


# if we are using a generative model with dense layers,
# we now preprocess and flatten the input
if flat_input:
    # add preprocessing steps here
    
    # flatten input
    img_train = img_train.reshape(img_train.shape[0], args.npx*args.npx)

    # apply pca transform
    if args.pca:
        scaler = StandardScaler()
        scaler.fit(img_train)
        img_train = scaler.transform(img_train)
        pca = PCA(0.95)
        pca.fit(img_train)
        img_train = pca.transform(img_train)
    
    if args.zca:
        scaler = StandardScaler()
        scaler.fit(img_train)
        img_train = scaler.transform(img_train)
        img_train, zca = zca_whiten(img_train)
else:
    # add preprocessing steps for full images (e.g. ZCA?)
    if args.zca:
        img_train = img_train.reshape(img_train.shape[0], args.npx*args.npx)

        scaler = StandardScaler()
        scaler.fit(img_train)
        img_train = scaler.transform(img_train)
        img_train, zca = zca_whiten(img_train)

        img_train = img_train.reshape(-1, args.npx, args.npx, 1)
    
print(img_train.shape)

# normalisation of images
#img_train = (img_train - np.average(img_train, axis=0))

# plt.imshow(np.average(img_train, axis=0).transpose(),
#            origin='lower', aspect='auto')
# plt.show()

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

# train on the images
model.train(img_train, epochs=args.epochs,
            batch_size = args.batch_size)

# now generate a test sample and save it
gen_sample = model.generate(args.ngen)

# invert the preprocessing/reshape to image if flattened
if flat_input:
    # undo preprocessing

    # reshape to a 2-d image
    if args.pca:
        gen_sample = scaler.inverse_transform(pca.inverse_transform(gen_sample)).reshape(args.ngen, args.npx, args.npx)
    elif args.zca:
        gen_sample = scaler.inverse_transform(np.dot(gen_sample, zca)).reshape(args.ngen, args.npx, args.npx)
    else:
        gen_sample = gen_sample.reshape(args.ngen, args.npx, args.npx)
else:
    # image processing
    if args.zca:
        gen_sample = scaler.inverse_transform(np.dot(gen_sample.reshape(-1, args.npx*args.npx), zca)).reshape(args.ngen, args.npx, args.npx)
    else:
        gen_sample = gen_sample.reshape(args.ngen, args.npx, args.npx)

np.save(args.output, gen_sample)
plt.imshow(np.average(gen_sample, axis=0))
plt.show()
