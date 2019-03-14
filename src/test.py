from read_data import Reader, Jets 
from JetTree import *
import matplotlib.pyplot as plt
from gan import GAN
from wgan_gp import WGANGP
from wgan import WGAN
import argparse
def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white

parser = argparse.ArgumentParser(description='Train a gan.')
parser.add_argument('--wgangp',action='store_true',dest='wgangp')
parser.add_argument('--wgan',action='store_true',dest='wgan')
parser.add_argument('--vae',action='store_true',dest='vae')
parser.add_argument('--nev', '-n', type=int, default=20000, help='Number of events.')
parser.add_argument('--npix', '-p', type=int, default=28, help='Number of pixels.')
args = parser.parse_args()

nev = args.nev
npxlx = args.npix
npxly = args.npix

reader=Jets('../../data/valid/valid_QCD_500GeV.json.gz',nev)
events=reader.values() 
lundimages=np.zeros((nev, npxlx, npxly, 1))
litest=[] 
li_gen=LundImage(npxlx = npxlx, npxly = npxly) 
for i, jet in enumerate(events): 
    tree = JetTree(jet) 
    lundimages[i]=li_gen(tree).reshape(npxlx, npxly, 1)

print(lundimages.shape)

# normalisation of images
#lundimages = (lundimages - np.average(lundimages, axis=0))

# plt.imshow(np.average(lundimages, axis=0).transpose(),
#            origin='lower', aspect='auto')
# plt.show()

if args.wgan:
    gan = WGAN(width=npxlx, height=npxly, channels=1)
    gan.train(lundimages, epochs=2000, batch_size=32, sample_interval=50)
if args.wgangp:
    gan = WGANGP(width=npxlx, height=npxly, channels=1)
    gan.train(lundimages, epochs=2000, batch_size=32, sample_interval=50)
elif args.vae:
    print("not implemented")
else:
    gan = GAN(width=npxlx, height=npxly, channels=1)
    gan.train(lundimages)
