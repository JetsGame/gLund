from read_data import Reader, Jets 
from JetTree import *
import matplotlib.pyplot as plt
from gan import GAN
from dcgan import DCGAN
from wgan_gp import WGANGP
from wgan import WGAN
import argparse

parser = argparse.ArgumentParser(description='Train a gan.')
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
parser.add_argument('--npx', type=int, default=28, help='Number of pixels.')
parser.add_argument('--data', type=str,
                    default='../../data/valid/valid_QCD_500GeV.json.gz',
                    help='Data set on which to train.')
parser.add_argument('--output', type=str, required=True, help='Output file.')
args = parser.parse_args()
# check that input is valid
if not (args.gan+args.dcgan+args.wgan+args.wgangp+args.vae == 1):
    raise ValueError('Invalid input: choose one model at a time.')
if args.gan or args.vae:
    # for GAN or VAE, we want to flatten the input and preprocess it
    flat_input = True
else:
    flat_input = False

# read in the data set
reader=Jets(args.data, args.nev)
events=reader.values() 
lundimages=np.zeros((args.nev, args.npx, args.npx, 1))
litest=[] 
li_gen=LundImage(npxlx = args.npx) 
for i, jet in enumerate(events): 
    tree = JetTree(jet) 
    lundimages[i]=li_gen(tree).reshape(args.npx, args.npx, 1)

# if we are using a generative model with dense layers,
# we now preprocess and flatten the input
if flat_input:
    # add preprocessing steps here
    
    # flatten input
    lundimages = lundimages.reshape(args.nev, args.npx*args.npx)
else:
    # add preprocessing steps for full images (e.g. ZCA?)
    pass
    
print(lundimages.shape)

# normalisation of images
#lundimages = (lundimages - np.average(lundimages, axis=0))

# plt.imshow(np.average(lundimages, axis=0).transpose(),
#            origin='lower', aspect='auto')
# plt.show()

# now set up the model
if args.wgan:
    model = WGAN(width=args.npx, height=args.npx)
    model.train(lundimages, epochs=args.epochs,
                batch_size=args.batch_size, sample_interval=50)
elif args.wgangp:
    model = WGANGP(width=args.npx, height=args.npx)
    model.train(lundimages, epochs=args.epochs,
                batch_size=args.batch_size, sample_interval=50)
elif args.vae:
    model = VAE(length=(args.npx*args.npx), mse_loss=False)
    model.train(lundimages, epochs=args.epochs,
                batch_size = args.batch_size)
elif args.dcgan:
    model = DCGAN(width=args.npx, height=args.npx)
    model.train(lundimages, epochs=args.epochs,
                batch_size = args.batch_size)
else:
    model = GAN(width=args.npx, height=args.npx)
    model.train(lundimages, epochs=args.epochs,
                batch_size = args.batch_size)

# now generate a test sample and save it
gen_sample = model.generate(args.ngen)

# invert the preprocessing/reshape to image if flattened
if flat_input:
    # undo preprocessing

    # reshape to a 2-d image
    gen_sample = gen_sample.reshape(args.ngen, args.npx, args.npx)
else:
    # image processing
    gen_sample = gen_sample.reshape(args.ngen, args.npx, args.npx)

np.save(args.output, gen_sample)
print(gen_sample.shape)
plt.imshow(np.average(gen_sample, axis=0))
plt.show()
